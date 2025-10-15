#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import struct
from collections import defaultdict
import os
import sys
import json
import open3d as o3d
import cv2
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import glob
from tqdm import tqdm
import shutil
from utils.transform_utils import project_points_to_depth
from scipy.spatial.transform import Rotation


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    is_test: bool
    K: np.array
    D: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

class FrameSceneListInfo(NamedTuple):
    point_cloud_list: list
    ply_path_list: str
    ply_path: str
    lidar_rotations: list
    lidar_translations: list
    train_cameras_list: list
    test_cameras_list: list
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    is_nerf_synthetic: bool
    pose_cl: np.ndarray

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)





def create_voxel_grid(all_points, s):
    # 计算全局包围盒并扩展边界
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0) + 1e-8  # 扩展边界处理浮点误差

    # 计算各维度体素数量
    min_x, min_y, min_z = min_coords
    max_x, max_y, max_z = max_coords
    num_x = int(np.ceil((max_x - min_x) / s))
    num_y = int(np.ceil((max_y - min_y) / s))
    num_z = int(np.ceil((max_z - min_z) / s))

    # 初始化体素网格
    voxel_grid = np.zeros((num_x, num_y, num_z), dtype=bool)

    # 计算每个点对应的体素索引
    indices_x = ((all_points[:, 0] - min_x) // s).astype(int)
    indices_y = ((all_points[:, 1] - min_y) // s).astype(int)
    indices_z = ((all_points[:, 2] - min_z) // s).astype(int)

    # 去重并标记占据的体素
    voxel_indices = np.column_stack((indices_x, indices_y, indices_z))
    unique_indices = np.unique(voxel_indices, axis=0)
    voxel_grid[tuple(unique_indices.T)] = True
    origin = np.array([min_x, min_y, min_z])
    occupied_indices = np.argwhere(voxel_grid)

    points = origin + (occupied_indices + 0.5) * s  # 计算体素中心
    return points



class Kitti360Loader:
    def __init__(self, root_path, root_image_path, sequence="0000"):
        self.root = root_path
        self.root_image = root_image_path
        self.sequence = sequence  # 格式如 "2013_05_28_drive_0000_sync"

        # 路径定义
        self.calib_path = os.path.join(root_path, "calibration")
        self.color_path = os.path.join(root_image_path, "data_2d_raw", self.sequence)
        self.velodyne_path = os.path.join(root_path, "data_3d_raw", self.sequence, 'velodyne_points', 'data')
        self.pose_path = os.path.join(root_path, "data_poses", self.sequence)

        # 加载标定参数和时间戳
        self.calib = self._load_calibration()
        self.timestamps = self._load_timestamps()
        self.all_poses, self.effective_indices = self.load_all_pose()
        self.shift_translation = 0.

    def _load_calibration(self):
        calib = {}

        perspective_path = os.path.join(self.calib_path, "perspective.txt")

        with open(perspective_path, 'r') as f:
            current_cam = None
            for line in f:
                line = line.strip()
                if not line or line.startswith(("calib_time", "corner_dist")):
                    continue  # 跳过注释和无关行

                # 解析键值对
                key, value = line.split(':', 1)
                key = key.strip()
                value = list(map(float, value.strip().split()))

                # 分离参数类型和相机ID (例如: K_00 -> ("K", "00"))
                if "_rect_" in key:  # 处理R_rect_00/P_rect_00等
                    param_type, cam_id = key.rsplit("_", 2)[0], key.split("_")[-1]
                    param_type = f"{param_type}_rect"
                else:
                    parts = key.split('_')
                    cam_id = parts[-1]
                    param_type = '_'.join(parts[:-1])  # 原始参数类型

                # 初始化相机存储结构
                cam_key = f"cam{cam_id}"
                if cam_key not in calib:
                    calib[cam_key] = {
                        "S": None, "K": None, "D": None,
                        "R": None, "T": None,
                        "S_rect": None, "R_rect": None, "P_rect": None
                    }

                # 赋值参数（按类型处理形状）
                if param_type == "S":
                    calib[cam_key]["S"] = np.array(value, dtype=np.int32)  # [width, height]
                elif param_type == "K":
                    calib[cam_key]["K"] = np.array(value).reshape(3, 3)  # 3x3内参矩阵
                    calib[cam_key]["K"][2,2] = 1.
                elif param_type == "D":
                    calib[cam_key]["D"] = np.array(value)  # 畸变系数k1,k2,p1,p2,k3
                elif param_type == "R":
                    calib[cam_key]["R"] = np.array(value).reshape(3, 3)  # 旋转矩阵
                elif param_type == "T":
                    calib[cam_key]["T"] = np.array(value)  # 平移向量
                elif param_type == "S_rect":
                    calib[cam_key]["S_rect"] = np.array(value, dtype=np.int32)
                elif param_type == "R_rect":
                    calib[cam_key]["R_rect"] = np.array(value).reshape(3, 3)
                elif param_type == "P_rect":
                    calib[cam_key]["P_rect"] = np.array(value).reshape(3, 4)

        # 构建相机外参矩阵 T_cam0_cam（相机->车体）
        for cam_id in calib:
            R = calib[cam_id]["R"]
            T = calib[cam_id]["T"]
            T_cam0_cam = np.eye(4)
            T_cam0_cam[:3, :3] = R
            T_cam0_cam[:3, 3] = T
            calib[cam_id]["T_cam0_cam"] = T_cam0_cam

        # 2. 加载相机到车体的外参 Tbc
        cam_to_pose_path = os.path.join(self.calib_path, "calib_cam_to_pose.txt")
        with open(cam_to_pose_path, 'r') as f:
            for line in f:
                if line.startswith("image"):
                    parts = line.split()
                    cam_id = 'cam' + parts[0][-3:-1]
                    matrix = np.array([float(x) for x in parts[1:]]).reshape(3, 4)
                    matrix_pose = np.eye(4)
                    matrix_pose[:3, :4] = matrix
                    calib[f"T_body_{cam_id}"] = matrix_pose  # 相机到车体

        # 3. 加载LiDAR到车体的变换 Tlc
        velo_to_body_path = os.path.join(self.calib_path, "calib_cam_to_velo.txt")
        with open(velo_to_body_path, 'r') as f:
            matrix = np.array([float(x) for x in f.read().split()]).reshape(3, 4)
            matrix_pose = np.eye(4)
            matrix_pose[:3, :4] = matrix
            calib["T_velo_cam00"] = matrix_pose  # 车体到LiDAR的变换
            T_body_cam0 = calib['T_body_cam00']
            T_body_velo = T_body_cam0 @ np.linalg.inv(matrix_pose)
            calib["T_body_velo"] = T_body_velo  # 车体到LiDAR的变换

        return calib

    def set_shift(self, shift_translation):
        self.shift_translation = shift_translation

    def _load_timestamps(self):
        timestamp_file = os.path.join(self.color_path, "image_00", "timestamps.txt")
        with open(timestamp_file, 'r') as f:
            return [line for line in f]

    def load_image(self, frame_id, camera="00", undistorted = False):
        frame_id = self.effective_indices[frame_id]
        frame_id = str(frame_id).zfill(10)
        img_folder = os.path.join(self.color_path, f"image_{camera}", 'data_rgb')
        img_path = os.path.join(img_folder, f"{frame_id}.png")
        img = np.array(Image.open(img_path))
        if undistorted:
            img = cv2.undistort(img, self.get_K(cam_id=camera), self.get_D(cam_id=camera))
        return img

    def get_image_path(self, frame_id, camera="00"):
        frame_id = self.effective_indices[frame_id]
        frame_id = str(frame_id).zfill(10)
        img_folder = os.path.join(self.color_path, f"image_{camera}", 'data_rgb')
        img_path = os.path.join(img_folder, f"{frame_id}.png")
        return img_path


    def load_pointcloud(self, frame_id):
        frame_id = self.effective_indices[frame_id]
        frame_id = str(frame_id).zfill(10)
        pc_path = os.path.join(self.velodyne_path, f"{frame_id}.bin")
        points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
        points = points[:, :3]
        points = points[np.logical_or(((np.abs(points[:, 0]) >= 1) & (np.abs(points[:, 1]) >= 1) & (points[:, 2] > -3)),
                                      ((points[:, 2] < -1)) & (points[:, 2] > -3))]
        return points

    def load_all_pose(self):
        poses = np.loadtxt(os.path.join(self.pose_path, "poses.txt"))
        effective_indices = (poses[:,0]).astype(np.int32)
        return poses, effective_indices

    def load_pose(self, frame_id):
        poses = self.all_poses

        pose_line = poses[poses[:, 0] == frame_id][0]

        matrix = pose_line[1:].reshape(3, 4)


        R = matrix[:3, :3]

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = matrix[:3,3]
        return T

    def project_lidar_to_camera(self, points, cam_id="00"):
        T_body_velo = self.calib["T_body_velo"]
        points_body = (T_body_velo @ self._homogenize(points[:, :3]).T).T

        T_body_cam = self.calib[f"T_body_cam{cam_id}"]
        points_cam = (np.linalg.inv(T_body_cam) @ points_body.T).T
        return points_cam[:, :3]

    def get_lidar_pose(self, frame_id):
        frame_id = self.effective_indices[frame_id]
        T_world_body = self.load_pose(frame_id)
        T_body_velo = self.calib["T_body_velo"]
        pose = T_world_body @ T_body_velo

        pose[:3, 3] -= self.shift_translation
        return pose

    def get_camera_pose(self, frame_id, cam_id="00"):
        frame_id = self.effective_indices[frame_id]
        T_world_body = self.load_pose(frame_id)
        T_body_cam = self.calib[f"T_body_cam{cam_id}"]
        pose = T_world_body @ (T_body_cam)
        pose[:3,3] -= self.shift_translation
        return pose

    def _homogenize(self, points):
        return np.hstack([points, np.ones((points.shape[0], 1))])

    def get_K(self, cam_id = '00'):
        return self.calib['cam' + cam_id]['K']


    def get_D(self, cam_id = '00'):
        return self.calib['cam' + cam_id]['D']

    def get_lidar_to_cam_pose(self, cam_id):
        Tbc = self.calib['T_body_cam' + cam_id]
        Tbl = self.calib['T_body_velo']
        Tcl = np.linalg.inv(Tbc) @ Tbl
        return Tcl

    def get_image_shape(self, cam_id = '00'):
        return int(self.calib['cam' + cam_id]['S'][0]), int(self.calib['cam' + cam_id]['S'][1])





def readKITTI360SceneInfo(root_path, root_image_path, sequence="00", cam_id='00', start_index = 0, segment_length = 10):
    kitti_loader = Kitti360Loader(
        root_path=root_path,
        root_image_path=root_image_path,
        sequence=sequence
    )

    cam_matrix = kitti_loader.get_K(cam_id=cam_id)
    D = kitti_loader.get_D(cam_id=cam_id)
    Tcl = kitti_loader.get_lidar_to_cam_pose(cam_id)
    fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]

    width, height = kitti_loader.get_image_shape(cam_id = cam_id)

    print("Generate initial point cloud...")
    all_points_world = []
    all_points_lidar = []
    all_lidar_rotations = []
    all_lidar_translations = []
    end_index = start_index + segment_length

    initial_translation = kitti_loader.get_lidar_pose(start_index)[:3,3]
    kitti_loader.set_shift(initial_translation)

    for frame_id in tqdm(range(start_index, end_index)):
        pc_lidar = kitti_loader.load_pointcloud(frame_id)
        T_lidar = kitti_loader.get_lidar_pose(frame_id)
        R_lidar = T_lidar[:3, :3]
        t_lidar = T_lidar[:3, 3:4]
        all_lidar_rotations.append(R_lidar)
        all_lidar_translations.append(t_lidar.reshape(-1))
        pc_world = (R_lidar @ pc_lidar[:, :3].T + t_lidar).T
        all_points_lidar.append(pc_lidar[:,:3])
        all_points_world.append(pc_world)
    all_points_world = np.concatenate(all_points_world, axis=0)

    cam_infos = []
    local_cam_infos = []
    for frame_id in tqdm(range(start_index, start_index+segment_length), desc="Loading camera data"):
        T_world2cam = np.linalg.inv(kitti_loader.get_camera_pose(frame_id, cam_id=cam_id))
        R = T_world2cam[:3, :3].T
        T = T_world2cam[:3, 3]

        FovY = focal2fov(fy, height)
        FovX = focal2fov(fx, width)

        img_path = kitti_loader.get_image_path(frame_id, camera=cam_id)
        cam_info = CameraInfo(
            uid=frame_id,
            R=R, T=T,
            FovY=FovY, FovX=FovX,
            image_path=img_path,
            image_name=f"{sequence}_{frame_id:06d}",
            width=width,
            height=height,
            is_test= False,
            K = cam_matrix,
            D = D)
        cam_infos.append(cam_info)
        local_cam_infos.append([cam_info])


    nerf_norm = getNerfppNorm([c for c in cam_infos])


    if not os.path.exists(os.path.join(root_path, 'ply')):
        os.makedirs(os.path.join(root_path, 'ply'))

    all_voxel_points = []
    all_voxel_points_world = all_points_world
    for index, point_cloud in tqdm(enumerate(all_points_lidar), desc='Processing point cloud'):
        all_voxel_points.append(point_cloud)

    shs = np.random.random((len(all_voxel_points_world), 3)) / 255.0
    colors = SH2RGB(shs) * 255
    global_ply_path = os.path.join(root_path, 'ply', f"model_global_point_cloud_{sequence}_{cam_id}.ply")
    storePly(global_ply_path, all_voxel_points_world, colors)
    global_pcd = fetchPly(global_ply_path)

    pcd_list = []
    ply_path_list = []
    for idnex, voxel_point_cloud in enumerate(all_voxel_points):
        shs = np.random.random((len(voxel_point_cloud), 3)) / 255.0
        colors = SH2RGB(shs) * 255
        ply_path = os.path.join(root_path, 'ply', f"model_{idnex}_point_cloud_{sequence}_{cam_id}.ply")
        storePly(ply_path, voxel_point_cloud, colors)
        pcd = fetchPly(ply_path)
        pcd_list.append(pcd)
        ply_path_list.append(ply_path)


    test_cameras = train_cameras = [c for c in cam_infos]
    test_cameras_list = train_cameras_list = [[c for c in l] for l in local_cam_infos]

    cameras_data = []
    for cam_info in cam_infos:
        rotation = cam_info.R.T
        position = cam_info.T
        position = -rotation.T @ position
        rotation = rotation.T
        rotation = rotation.tolist()
        position = position.tolist() 

        img_name = os.path.basename(cam_info.image_path)

        camera_dict = {
            "id": int(cam_info.uid),
            "img_name": img_name,
            "width": int(cam_info.width),
            "height": int(cam_info.height),
            "fx": float(fx),
            "fy": float(fy),
            "position": [float(position[0]), float(position[1]), float(position[2])],
            "rotation": [
                [float(rotation[0][0]), float(rotation[0][1]), float(rotation[0][2])],
                [float(rotation[1][0]), float(rotation[1][1]), float(rotation[1][2])],
                [float(rotation[2][0]), float(rotation[2][1]), float(rotation[2][2])]
            ]
        }
        cameras_data.append(camera_dict)

    cameras_json_path = os.path.join(root_path, "cameras.json")
    with open(cameras_json_path, "w", encoding="utf-8") as f:
        json.dump(cameras_data, f, indent=4, ensure_ascii=False)


    frame_scene_list = FrameSceneListInfo(
        point_cloud_list=pcd_list,
        lidar_rotations=all_lidar_rotations,
        lidar_translations=all_lidar_translations,
        train_cameras=train_cameras,
        train_cameras_list=train_cameras_list,
        test_cameras=test_cameras,
        test_cameras_list=test_cameras_list,
        nerf_normalization=nerf_norm,
        ply_path_list= ply_path_list,
        ply_path=global_ply_path,
        is_nerf_synthetic=False,
        pose_cl = Tcl,
    )

    return frame_scene_list, all_voxel_points_world



sceneLoadTypeCallbacks = {
    "KITTI360": readKITTI360SceneInfo,
}