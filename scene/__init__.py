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

import os
import random
import json
import pytorch3d
import pytorch3d.transforms
import cv2

from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.calibrate_gaussian_model import CalibrateGaussianModel, CalibrateGaussianListModel
from sensors.sensor_trajectories import SensorTrajectories
import numpy as np
from romatch import roma_outdoor
import torch
import torch.nn.functional as F


class Scene:

    gaussians : CalibrateGaussianListModel

    def __init__(self, args : ModelParams, gaussians : CalibrateGaussianListModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], voxel_resolutions = [0.3], sensor_trajectories=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.weight_path = args.flow_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.resolution_scales = resolution_scales
        self.voxel_resolutions = voxel_resolutions
        self.current_level = 0


        # Only support KITTI now
        if os.path.exists(os.path.join(args.source_path, "data_3d_raw")):
            dataset_type = 'KITTI360'
            base_path = args.source_path
            image_path = base_path
            start_indices = [897, 2907, 86, 10948, 2743]
            scene_ids = ['1', '2', '3', '4', '5']
            start_index = start_indices[args.data_seq]
            scene_id = scene_ids[args.data_seq]
            cam_id = args.cam_id
            scene_info_list, all_lidar_points = sceneLoadTypeCallbacks["KITTI360"](root_path = args.source_path, root_image_path = image_path, sequence=scene_id,
                                                                                cam_id=cam_id, start_index=start_index,
                                                                                segment_length=30)
        else:
            assert False, "Could not recognize scene type!"

        self.scene_info_list = scene_info_list
        self.all_lidar_points = all_lidar_points

        if not self.loaded_iter:
            with open(scene_info_list.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info_list.test_cameras:
                camlist.extend(scene_info_list.test_cameras)
            if scene_info_list.train_cameras:
                camlist.extend(scene_info_list.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        self.cameras_extent = scene_info_list.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info_list.train_cameras, resolution_scale, args, scene_info_list.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info_list.test_cameras, resolution_scale, args, scene_info_list.is_nerf_synthetic, True)


        #Generate flow model, the resolution should be the finest one
        self.generate_optical_flow()

        initialized_pose_cl = None
        if dataset_type == 'KITTI360' or 'KITTI':
            initial_rotation_mat = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
            initialized_pose_cl = np.eye(4)
            initialized_pose_cl[:3,:3]  = initial_rotation_mat

        # Generate trajectory
        self.sensor_trajectory = SensorTrajectories(
            lidar_rotations=scene_info_list.lidar_rotations,
            lidar_translations=scene_info_list.lidar_translations,
            lidar_timestamps=[0. for i in range(len(scene_info_list.lidar_rotations))],
            pose_cl=scene_info_list.pose_cl,
            initialized_pose_cl=initialized_pose_cl
        )



        for resolution_scale in resolution_scales:
            for i in range(len(self.train_cameras[resolution_scale])):
                cam = self.train_cameras[resolution_scale][i]
                # latter_cam = self.train_cameras[resolution_scale][i+1]
                cam.generate_gradient_and_pixel_masks()
            for i in range(len(self.test_cameras[resolution_scale])):
                cam = self.test_cameras[resolution_scale][i]
                # latter_cam = self.test_cameras[resolution_scale][i + 1]
                cam.generate_gradient_and_pixel_masks()


        rotation_wl_list, translations_wl_list = self.sensor_trajectory.get_all_lidar_poses()
        projection_matrix = self.train_cameras[resolution_scales[self.current_level]][0].projection_matrix

        self.gaussians.create_from_pcd_list(pcd_list=scene_info_list.point_cloud_list,
                                            spatial_lr_scale= self.cameras_extent,
                                            rotations_wl_list= rotation_wl_list,
                                            translations_wl_list=translations_wl_list,
                                            projection_matrix = projection_matrix,
                                            resolution=self.voxel_resolutions[self.current_level])

    def generate_optical_flow(self):
        #Generate the optical flow on the finest resolution
        resolution_scale = self.resolution_scales[-1]
        train_cameras = self.train_cameras[resolution_scale]

        dino_weights = torch.load(os.path.join(self.weight_path, 'dinov2_vitl14_pretrain.pth'))
        weights = torch.load(os.path.join(self.weight_path, 'roma_outdoor.pth'))
        H = train_cameras[0].image_height
        W = train_cameras[0].image_width
        flow_model = roma_outdoor(device='cuda', weights=weights, dinov2_weights=dino_weights, coarse_res=560,
                                       upsample_res=(H, W))

        def generate_optical_flow_one_group(flow_model, cameras, window_size = 2):
            cam_num = len(cameras)
            images = [(camera.original_image.detach().cpu().permute(1,2,0).numpy()*255.0).astype(np.uint8) for camera in cameras]
            all_flow_refs = {}
            all_flow_matchs = {}
            all_certainty_refs = {}
            all_certainty_matchs = {}
            print('Start to generate optical flow')
            for w in range(1, window_size+1):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print('Window size', w)
                flow_refs = []
                flow_matchs = []
                certainty_refs = []
                certainty_matchs = []
                for i in range(cam_num-w):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print('Generating optical flow for camera %d' % i)
                    ref_image = images[i]
                    match_image = images[i+w]
                    warp, certainty = flow_model.match_image(ref_image, match_image, device='cuda')
                    flow_ref = warp[:, :W, 2:]
                    flow_match = warp[:, W:, :2]
                    certainty_ref = certainty[:, :W]
                    certainty_match = certainty[:, W:]
                    flow_refs.append(flow_ref.detach().cpu())
                    flow_matchs.append(flow_match.detach().cpu())
                    certainty_refs.append(certainty_ref.detach().cpu())
                    certainty_matchs.append(certainty_match.detach().cpu())
                all_flow_refs[w] = flow_refs
                all_flow_matchs[w] = flow_matchs
                all_certainty_refs[w] = certainty_refs
                all_certainty_matchs[w] = certainty_matchs

            return all_flow_refs, all_flow_matchs, all_certainty_refs, all_certainty_matchs

        #Get the flow on the finest resolution
        window_size = 1
        all_flow_refs, all_flow_matchs, all_certainty_refs, all_certainty_matchs = generate_optical_flow_one_group(flow_model, train_cameras, window_size=window_size)
        del flow_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for resolution_scale in list(self.train_cameras.keys()):
            cameras = self.train_cameras[resolution_scale]
            #Have already been computed
            level_width, level_height = cameras[0].image_width, cameras[0].image_height
            for w in range(1, window_size+1):
                flow_refs = all_flow_refs[w]
                flow_matchs = all_flow_matchs[w]
                certainty_refs = all_certainty_refs[w]
                certainty_matchs = all_certainty_matchs[w]

                flow_ref_level = [item.to(device = 'cuda') for item in flow_refs]
                flow_match_level = [item.to(device = 'cuda') for item in flow_matchs]
                certainty_ref_level = [item.to(device = 'cuda') for item in certainty_refs]
                certainty_match_level = [item.to(device = 'cuda') for item in certainty_matchs]

                #Rescale
                for i in range(len(flow_refs)):
                    flow_ref, flow_match, certainty_ref, certainty_match = flow_ref_level[i], flow_match_level[i], certainty_ref_level[i], certainty_match_level[i]
                    flow_ref_level[i] = F.interpolate(flow_ref[None, :, :, :].permute(0,3,1,2), size=(level_height, level_width),
                                                      mode='nearest').squeeze(0).permute(1,2,0)
                    flow_match_level[i] = F.interpolate(flow_match[None, :, :, :].permute(0,3,1,2), size=(level_height, level_width),
                                                      mode='nearest').squeeze(0).permute(1,2,0)
                    certainty_ref_level[i] = F.interpolate(certainty_ref[None, None, :, :], size=(level_height, level_width),
                                                      mode='nearest').squeeze(0).permute(1,2,0)
                    certainty_match_level[i] = F.interpolate(certainty_match[None, None, :, :], size=(level_height, level_width),
                                                      mode='nearest').squeeze(0).permute(1,2,0)

                #Bind flow
                cam_num = len(cameras)
                assert (cam_num == (len(flow_ref_level)+w)) & (cam_num == (len(flow_match_level)+w)) & (cam_num == (len(certainty_ref_level)+w)) & (cam_num == (len(certainty_match_level)+w))
                for i in range(cam_num-w):
                    cameras[i].forward_flows.append(flow_ref_level[i])
                    cameras[i].forward_flow_uids.append(i+w)
                    cameras[i].forward_certainties.append(certainty_ref_level[i])

                    cameras[i+w].backward_flows.append(flow_match_level[i])
                    cameras[i+w].backward_flow_uids.append(i)
                    cameras[i+w].backward_certainties.append(certainty_match_level[i])
            print('Finish generate optical flow')

    
    def update_level(self, level):
        self.current_level = level
        voxel_resolution = self.voxel_resolutions[self.current_level]
        
    #TODO: Not implemented yet.
    def save(self, iteration):
        return
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTrainImages(self, scale = 1.0):
        train_images = []
        for camera in self.train_cameras[scale]:
            image = camera.original_image.detach().cpu().permute(1,2,0).numpy()
            image *= 255
            train_images.append(image.astype(np.uint8))
        return train_images

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getTrainCameraByIndex(self, index, scale=1.0):
        return self.train_cameras[scale][index]
