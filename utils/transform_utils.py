import torch
import numpy as np


def project_points_to_depth(camera_pose, intrinsics, points_3d, image_shape):
    """
    将3D点云投影至相机坐标系生成深度图

    参数：
    - camera_pose : 4x4相机外参矩阵 [[R, t], [0, 1]]，将点从世界坐标变换到相机坐标
    - intrinsics  : 3x3相机内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    - points_3d   : Nx3的3D点云，世界坐标系
    - image_shape : 输出深度图的尺寸 (height, width)

    返回：
    - depth_map : 深度图，未投影区域填充0
    - mask      : 布尔掩码，标记有效投影区域
    """
    H, W = image_shape

    # 1. 坐标变换至相机坐标系
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    points_hom = np.hstack((points_3d, np.ones((len(points_3d), 1))))  # 齐次坐标
    points_cam = (camera_pose @ points_hom.T).T[:, :3]  # 应用外参变换

    # 2. 过滤相机后方的点 (Z <= 0)
    valid = points_cam[:, 2] > 0
    points_cam = points_cam[valid]
    if points_cam.size == 0:
        return np.zeros((H, W)), np.zeros((H, W), dtype=bool)

    # 3. 投影至图像平面
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # 归一化坐标 (注意避免除零)
    z = points_cam[:, 2]
    x = points_cam[:, 0] / z
    y = points_cam[:, 1] / z

    # 像素坐标 (四舍五入取整)
    u = np.round(fx * x + cx).astype(int)
    v = np.round(fy * y + cy).astype(int)

    # 4. 过滤超出边界的点
    valid_pixels = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[valid_pixels], v[valid_pixels], z[valid_pixels]
    if z.size == 0:
        return np.zeros((H, W)), np.zeros((H, W), dtype=bool)

    # 5. 生成深度图和掩码
    depth_map = np.full((H, W), np.inf)  # 初始化无穷大
    mask = np.zeros((H, W), dtype=np.float32)

    # 使用最小深度更新 (处理多个点投影到同一像素)
    np.minimum.at(depth_map, (v, u), z)

    # 生成有效掩码
    mask[v, u] = 1.
    depth_map[mask != 1.] = 1e-7  # 无效区域置零

    return depth_map, mask