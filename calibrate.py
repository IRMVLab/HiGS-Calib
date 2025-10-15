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
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import cv2
import numpy as np
import pytorch3d.transforms
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, weighted_l1_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, CalibrateGaussianListModel
from utils.general_utils import safe_state, get_expon_lr_func, sample_uniform
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from visualizer.pose_visualizer import *
from visualizer.image_visualizer import project_lidar_to_image_with_projection, \
    project_to_pixel_torch, get_3d_points_from_pixels_depth_mask_torch
import torch.nn.functional as F
import time

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False



def dilate_false_mask(mask: torch.Tensor, window_size: int) -> torch.Tensor:
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd number, e.g., 3,5,7...")

    assert mask.dtype == torch.bool, "Input mask must be a boolean tensor."

    inverted_mask = ~mask
    inverted_mask_float = inverted_mask.float()

    kernel = torch.ones(
        (1, 1, window_size, window_size),
        dtype=torch.float32,
        device=mask.device
    )

    shape = inverted_mask_float.shape
    if inverted_mask_float.dim() == 2:  # (H, W) → (1, 1, H, W)
        inverted_mask_float = inverted_mask_float.view(1, 1, shape[0], shape[1])
    elif inverted_mask_float.dim() == 3:  # (C, H, W) → (1, C, H, W)
        inverted_mask_float = inverted_mask_float.unsqueeze(0)

    padding = window_size // 2
    conved = F.conv2d(inverted_mask_float, kernel, padding=padding)

    dilated_inverted = conved > 0

    dilated_inverted = dilated_inverted.view(shape)

    result = ~dilated_inverted
    return result


def compute_rgb_projection_error(
        rgb_render,
        rgb_target,
        depth_render,
        depth_mask,
        gradient_mask,
        points3d_world,
        rotation_wc_ref,
        translation_wc_ref,
        rotation_wc_target,
        translation_wc_target,
        P,
        target_depth = None):
    img_height, img_width = rgb_render.shape[1],rgb_render.shape[2]

    points3d_target = points3d_world @ rotation_wc_target  - (rotation_wc_target.T @ translation_wc_target)
    project_depth = points3d_target[:,2]

    target2d = project_to_pixel_torch(points3d_target, P, img_width, img_height)
    us,vs = target2d[:,0], target2d[:,1]
    projection_mask = (target2d[:,0] >= 0.) & (target2d[:,0] < img_width) & (target2d[:,1] >= 0.) & (target2d[:,1] < img_height)
    us = us[projection_mask]/img_width
    vs = vs[projection_mask]/img_height
    grid_x = us * 2 - 1  # [0,1] → [-1,1]
    grid_y = vs * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (N, 2)
    grid = grid.view(1, 1, -1, 2)


    # 双线性插值采样
    sampled = F.grid_sample(
        rgb_target.unsqueeze(0),
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(0).squeeze(1)  # (1, C, 1, N)
    sampled_depth = F.grid_sample(
        target_depth.unsqueeze(0),
        grid,
        mode='nearest',
        padding_mode='border',
        align_corners=True
    ).squeeze(0).squeeze(1)
    #Inverse depth
    project_depth_diff = project_depth[projection_mask] - 1. / sampled_depth.squeeze(0)
    # project_depth_diff_ratio = project_depth_diff * sampled_depth.squeeze(0)
    depth_culling = ((project_depth_diff) < 0.2)
    indices_true = torch.where(depth_mask.flatten())[0][projection_mask][depth_culling]

    sampled = sampled[:,depth_culling]
    initial_rgb = rgb_render.view(3,-1)[:,indices_true]
    weights = (2 - gradient_mask[:,depth_mask.squeeze(0)][:,projection_mask][:,depth_culling]).detach()

    return torch.mean( weights * torch.abs(sampled - initial_rgb))



def compute_rgb_projection_error_and_flow_error(
        rgb_render,
        rgb_target,
        depth_render,
        depth_mask,
        gradient_mask,
        points3d_world,
        rotation_wc_ref,
        translation_wc_ref,
        rotation_wc_target,
        translation_wc_target,
        P,
        flow,
        flow_certainties,
        visualize = True,
        target_depth = None
        ):
    img_height, img_width = rgb_render.shape[1],rgb_render.shape[2]


    points3d_target = points3d_world @ rotation_wc_target  - (rotation_wc_target.T @ translation_wc_target)
    project_depth = points3d_target[:, 2]

    target2d = project_to_pixel_torch(points3d_target, P, img_width, img_height)
    us,vs = target2d[:,0], target2d[:,1]
    projection_mask = (target2d[:,0] >= 0.) & (target2d[:,0] < img_width) & (target2d[:,1] >= 0.) & (target2d[:,1] < img_height)
    us = us[projection_mask]/img_width
    vs = vs[projection_mask]/img_height
    grid_x = us * 2 - 1  # [0,1] → [-1,1]
    grid_y = vs * 2 - 1

    projection_pos = torch.concatenate([grid_x[:,None], grid_y[:,None]], dim=1)


    #Compute flow error
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (N, 2)
    # 添加批次和采样点维度 → (1, 1, N, 2)
    grid = grid.view(1, 1, -1, 2)


    # 双线性插值采样
    sampled = F.grid_sample(
        rgb_target.unsqueeze(0),
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(0).squeeze(1)  # (1, C, 1, N)
    sampled_depth = F.grid_sample(
        target_depth.unsqueeze(0),
        grid,
        mode='nearest',
        padding_mode='border',
        align_corners=True
    ).squeeze(0).squeeze(1)


    # Inverse depth
    project_depth_diff = project_depth[projection_mask] - 1. / sampled_depth.squeeze(0)
    # project_depth_diff_ratio = project_depth_diff * sampled_depth.squeeze(0)
    depth_culling = ((project_depth_diff) < 0.2)


    indices_true = torch.where(depth_mask.flatten())[0][projection_mask][depth_culling]
    # flow_masked = flow[depth_mask.squeeze(0)]
    # flow_certainties_masked = flow_certainties[depth_mask.squeeze(0)]

    flow_pos = flow.view(-1,2)[indices_true]
    if len(flow_pos) == 0:
        flow_loss = 0.
    else:
        flow_mask = (flow_certainties.view(-1,1)[indices_true] > 0.5).squeeze(1)
        # flow_certainties = flow_certainties_masked[projection_mask][depth_culling].squeeze(1)[flow_mask]
        # flow_loss = torch.mean((1 + flow_certainties).unsqueeze(1) * torch.abs(flow_pos[flow_mask] - projection_pos[depth_culling][flow_mask]))
        flow_loss = torch.mean(torch.abs(flow_pos[flow_mask] - projection_pos[depth_culling][flow_mask]))

    sampled = sampled[:,depth_culling]

    initial_rgb = rgb_render.view(3, -1)[:, indices_true]
    weights = (2 - gradient_mask[:, depth_mask.squeeze(0)][:, projection_mask][:, depth_culling]).detach()

    match_loss = torch.mean(weights * torch.abs(sampled - initial_rgb))

    return match_loss, flow_loss


def compute_depth_error(point_cloud_lidar, scene, P, depth_render, depth_mask):
    rotation_cl, translation_cl = scene.sensor_trajectory.get_extrinsics()
    rotation_cl = rotation_cl.detach()
    rotation_cl = pytorch3d.transforms.quaternion_to_matrix(rotation_cl)
    translation_cl = translation_cl.detach() * 0.
    point_cloud_camera = point_cloud_lidar @ rotation_cl.T + translation_cl
    point_cloud_depths = point_cloud_camera[:, 2]
    width = depth_render.shape[2]
    height = depth_render.shape[1]
    point_cloud_pixels = project_to_pixel_torch(point_cloud_camera, P, width, height)
    u, v = point_cloud_pixels[:, 0], point_cloud_pixels[:, 1]
    valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (point_cloud_depths > 0.1) & (
                point_cloud_depths < 100.)
    depth_render_view = depth_render.view(1, 1, height, width)
    grid_u = (u / (width - 1)) * 2 - 1  
    grid_v = (v / (height - 1)) * 2 - 1  
    grid = torch.stack([grid_u[valid_uv], grid_v[valid_uv]], dim=1).view(1, -1, 1, 2)  # [1,N,1,2]
    depth_sampled = F.grid_sample(depth_render_view, grid, mode='nearest', align_corners=True).view(-1)
    depth_mask_loss = F.grid_sample(depth_mask.float().unsqueeze(0), grid, mode='nearest', align_corners=True).view(-1)
    depth_loss = l1_loss(depth_sampled * depth_mask_loss, 1. / point_cloud_depths[valid_uv] * depth_mask_loss)
    return depth_loss




def compute_match_loss(scene, vind, all_depths, all_depth_masks, resolution_scale, gt_image, gt_gradient_mask, gt_pixel_mask, rendered_depth, depth_mask, window_size = 2, use_flow = False, visualize = False):
    match_loss = 0.
    loss_num = 0
    rotation_wc_ref, translation_wc_ref = scene.sensor_trajectory.get_camera_pose(vind)
    cam_num = len(scene.getTrainCameras())
    shift_indices = [i for i in range(-window_size, window_size+1) if vind+i >=0 and vind+i<cam_num and i is not 0]
    ref_cam = scene.getTrainCameraByIndex(vind, scale=resolution_scale)

    for shift in shift_indices:
        target_vind = vind + shift
        target_cam = scene.getTrainCameraByIndex(target_vind, scale=resolution_scale)
        depth_P = target_cam.projection_matrix.transpose(0, 1)
        rotation_wc_match, translation_wc_match = scene.sensor_trajectory.get_camera_pose(target_vind)
        rgb_target = target_cam.original_image.cuda()


        with torch.no_grad():
            target_depth = all_depths[target_vind]
            target_depth_mask = all_depth_masks[target_vind]
            target_depth[target_depth_mask == False] = 1e7

        flow = None
        flow_certainties = None
        if use_flow:
            for i in range(len(ref_cam.forward_flow_uids)):
                if target_vind == ref_cam.forward_flow_uids[i]:
                    flow = ref_cam.forward_flows[i]
                    flow_certainties = ref_cam.forward_certainties[i]
            for i in range(len(ref_cam.backward_flow_uids)):
                if target_vind == ref_cam.backward_flow_uids[i]:
                    flow = ref_cam.backward_flows[i]
                    flow_certainties = ref_cam.backward_certainties[i]

        img_height, img_width = gt_image.shape[1], gt_image.shape[2]
        points3d = get_3d_points_from_pixels_depth_mask_torch(
            depth_map=1. / rendered_depth.squeeze(0),
            mask=depth_mask.squeeze(0),
            P=depth_P,
            image_width=img_width,
            image_height=img_height)
        if len(points3d) == 0:
            return match_loss
        points3d_world = points3d @ rotation_wc_ref.T + translation_wc_ref

        if flow is None:
            rgb_rep_loss = compute_rgb_projection_error(
                rgb_render=gt_image,
                rgb_target=rgb_target,
                depth_render=rendered_depth,
                depth_mask=depth_mask,
                gradient_mask = gt_gradient_mask.unsqueeze(0),
                points3d_world=points3d_world,
                rotation_wc_ref=rotation_wc_ref,
                translation_wc_ref=translation_wc_ref,
                rotation_wc_target=rotation_wc_match,
                translation_wc_target=translation_wc_match,
                P=depth_P,
                target_depth=target_depth)
            flow_loss = None
        else:
            rgb_rep_loss, flow_loss = compute_rgb_projection_error_and_flow_error(
                rgb_render=gt_image,
                rgb_target=rgb_target,
                depth_render=rendered_depth,
                depth_mask=depth_mask,
                gradient_mask = gt_gradient_mask.unsqueeze(0),
                points3d_world=points3d_world,
                rotation_wc_ref=rotation_wc_ref,
                translation_wc_ref=translation_wc_ref,
                rotation_wc_target=rotation_wc_match,
                translation_wc_target=translation_wc_match,
                P=depth_P,
                flow=flow,
                flow_certainties=flow_certainties,
                visualize = visualize,
                target_depth=target_depth
            )
        match_loss += rgb_rep_loss
        if flow_loss is not None:
            match_loss += flow_loss
        loss_num += 1

    match_loss /= loss_num


    return match_loss


def generate_all_depths(scene, pipe, background, resolution_scale, train_test_exp, full_cut_size = 100, min_depth = 0.1, max_depth = 50.):
    all_depths = []
    all_depth_masks = []
    with torch.no_grad():
        train_cameras = scene.getTrainCameras(scale=resolution_scale).copy()
        cam_num = len(train_cameras)
        for i in range(cam_num):
            target_vind = i
            target_cam = train_cameras[target_vind]
            rotation_wc_match, translation_wc_match = scene.sensor_trajectory.get_camera_pose(target_vind)
            render_pkg = render(target_cam, scene.gaussians, pipe, background, use_trained_exp= train_test_exp,
                                    separate_sh=SPARSE_ADAM_AVAILABLE, cam_rotation=rotation_wc_match,
                                    cam_translation=translation_wc_match)
            target_depth = render_pkg['depth'].detach()

            target_depth_mask = torch.ones_like(target_depth).bool().detach()
            cut_region = int(full_cut_size / resolution_scale)
            target_depth_mask[:, :cut_region, :] = False
            inv_depth = 1. / target_depth
            target_depth_mask[inv_depth > max_depth] = False
            target_depth_mask[inv_depth < min_depth] = False
            all_depths.append(target_depth)
            all_depth_masks.append(target_depth_mask)
    return all_depths, all_depth_masks


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(
            f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    resolution_scales = [4., 2., 1., 1.]
    voxel_resolutions = [0.1, 0.1, 0.1, 0.1]
    initial_resolution = resolution_scales[0]
    change_iteration = [0, 2000, 4000, 8000]
    per_level_iterations = [2000, 2000, 4000, 7000]
    lr_ratio = [1., 0.5, 0.5, 0.1]
    current_scale_index = 0
    match_loss_window_size = 2
    fast_mode = False

    gaussians = CalibrateGaussianListModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, resolution_scales=resolution_scales, voxel_resolutions=voxel_resolutions)
    gaussians.training_setup(opt)
    scene.sensor_trajectory.training_setup(opt)

    # Initialize visualizer

    # 读取参数
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    bg_color = np.random.rand(3)
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE

    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    viewpoint_stack = scene.getTrainCameras(initial_resolution).copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    cut_full_size = 10
    min_depth = 0.1
    max_depth = 50.

    depth_view = None
    #Prepare depth maps for latter
    all_depths, all_depth_masks = generate_all_depths(scene, pipe, background, resolution_scales[current_scale_index], train_test_exp=dataset.train_test_exp, full_cut_size = cut_full_size, min_depth = min_depth, max_depth = max_depth)

    sum_iterations = 10000
    if fast_mode:
        sum_iterations = 6000
    # sum_iterations = 2000

    for iteration in range(first_iter, sum_iterations):
        start_time = time.perf_counter()
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer,
                                       use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)[
                        "render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()


        # Every 1000 its we increase the levels of SH up to a maximum degree
        # TODO:Not need now
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        if iteration >= change_iteration[1] and current_scale_index == 0:
            print('Change to pyramid level 1')
            current_scale_index = 1
            scene.update_level(current_scale_index)
            all_depths, all_depth_masks = generate_all_depths(scene, pipe, background, resolution_scales[current_scale_index], train_test_exp=dataset.train_test_exp, full_cut_size = cut_full_size, min_depth = min_depth, max_depth = max_depth)
            viewpoint_stack.clear()
            viewpoint_indices.clear()
            #Update learning rate
            scene.sensor_trajectory.update_learning_rate(lr_ratio[current_scale_index])
            print('Current point size: ', len(gaussians._xyz))
        elif iteration >= change_iteration[2] and current_scale_index == 1:
            print('Change to pyramid level 2')
            current_scale_index = 2
            scene.update_level(current_scale_index)
            all_depths, all_depth_masks = generate_all_depths(scene, pipe, background,
                                                              resolution_scales[current_scale_index],
                                                              train_test_exp=dataset.train_test_exp,
                                                              full_cut_size=cut_full_size, min_depth=min_depth,
                                                              max_depth=max_depth)
            viewpoint_stack.clear()
            viewpoint_indices.clear()
            #Update learning rate
            scene.sensor_trajectory.update_learning_rate(lr_ratio[current_scale_index])
            print('Current point size: ', len(gaussians._xyz))
        elif iteration >= change_iteration[3] and current_scale_index == 2:
            print('Change to pyramid level 3')
            current_scale_index = 3
            scene.update_level(current_scale_index)
            all_depths, all_depth_masks = generate_all_depths(scene, pipe, background,
                                                              resolution_scales[current_scale_index],
                                                              train_test_exp=dataset.train_test_exp,
                                                              full_cut_size=cut_full_size, min_depth=min_depth,
                                                              max_depth=max_depth)
            viewpoint_stack.clear()
            viewpoint_indices.clear()
            #Update learning rate
            scene.sensor_trajectory.update_learning_rate(lr_ratio[current_scale_index])
            print('Current point size: ', len(gaussians._xyz))

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(scale=resolution_scales[current_scale_index]).copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        rand_idx = randint(0, len(viewpoint_indices) - 1)
        # 每个迭代从训练视角集合中随机选取一个相机，确保模型多角度学习。
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
        cam_rotation, cam_translation = scene.sensor_trajectory.get_camera_pose(vind)


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background


        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE, cam_rotation=cam_rotation,
                            cam_translation=cam_translation)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]
        rendered_depth = render_pkg['depth']

        end_time_render = time.perf_counter()
        execution_time_render = end_time_render - start_time

        depth_mask = torch.ones_like(rendered_depth).bool().detach()
        cut_region = int(cut_full_size / resolution_scales[current_scale_index])
        depth_mask[:, :cut_region, :] = False
        image_mask = (depth_mask.repeat(3, 1, 1)).detach()
        inv_depth = 1. / rendered_depth.detach()
        depth_mask[inv_depth > max_depth] = False
        depth_mask[inv_depth < min_depth] = False

        # rendered_depth *= image_mask
        # Only use the bottom region of the image
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_gradient_mask = viewpoint_cam.gradient_mask.cuda()
        gt_pixel_mask = viewpoint_cam.pixel_mask.cuda()
        weights = gt_gradient_mask.unsqueeze(0).repeat(3,1,1)[image_mask]
        if scene.sensor_trajectory.rotation_cl_delta.requires_grad:
            Ll1 = weighted_l1_loss(image[image_mask], gt_image[image_mask], 2-weights)
        else:
            Ll1 = l1_loss(image[image_mask], gt_image[image_mask])

        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image, window_size=5)

        point_cloud_lidar = gaussians.gaussian_model_list[vind].original_point_cloud
        depth_P = viewpoint_cam.projection_matrix.transpose(0, 1)


        end_time_loss_render = time.perf_counter()
        execution_time_loss_render = end_time_loss_render - start_time

        _, lidar_translation = scene.sensor_trajectory.get_lidar_pose(vind)
        depth_cam_rotation = cam_rotation.detach()
        depth_lidar_translation = lidar_translation.detach()
        render_pkg_depth = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp,
                              separate_sh=SPARSE_ADAM_AVAILABLE, cam_rotation= depth_cam_rotation,
                              cam_translation=depth_lidar_translation)

        depth_mask2 = torch.ones_like(render_pkg_depth['depth']).bool().detach()
        depth_mask2[:, :cut_region, :] = False
        inv_depth2 = 1. / render_pkg_depth['depth'].detach()
        depth_mask2[inv_depth2 > max_depth] = False
        depth_mask2[inv_depth2 < min_depth] = False
        depth_loss = compute_depth_error(point_cloud_lidar, scene, depth_P, render_pkg_depth['depth'],
                                             depth_mask2.detach())


        end_time_loss_depth = time.perf_counter()
        execution_time_loss_depth = end_time_loss_depth - start_time
        match_loss = 0.
        if scene.sensor_trajectory.rotation_cl_delta.requires_grad and current_scale_index<3:
            visualize_match = False
            match_loss = compute_match_loss(
                scene=scene,
                vind=vind,
                all_depths=all_depths,
                all_depth_masks = all_depth_masks,
                resolution_scale=resolution_scales[current_scale_index],
                gt_image=gt_image,
                gt_gradient_mask=gt_gradient_mask,
                gt_pixel_mask = gt_pixel_mask,
                rendered_depth=rendered_depth,
                depth_mask=depth_mask,
                window_size=match_loss_window_size,
                use_flow=True,
                visualize = visualize_match)


        end_time_loss_match = time.perf_counter()
        execution_time_loss_match = end_time_loss_match - start_time

        loss = depth_loss * 10
        loss += match_loss
        if (not scene.sensor_trajectory.rotation_cl_delta.requires_grad  ) or current_scale_index >=3:
            loss += ((1.0 - opt.lambda_dssim) * Ll1)
            loss += opt.lambda_dssim * (1.0 - ssim_value)

        # Scale norm loss
        observable_mask = radii > 0.
        visible_scaling = gaussians.get_scaling[observable_mask]
        scale_constraint = torch.max(visible_scaling, dim=1).values / (torch.min(visible_scaling, dim=1).values + 1e-7)
        clip_constraint = torch.clip(scale_constraint - 10, min=0.)
        scaling_loss = torch.mean(clip_constraint)
        loss += 3e-2 * scaling_loss
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # if iteration % 10 == 0:
            #     img_index = 5
            #     if vind == img_index:
            #         depth_view = torch.clamp(render_pkg["depth"].detach().cpu(), 0.0, 1.0) * 4
            #         depth_view[torch.logical_not(depth_mask)] = 0.
            #         depth_view = depth_view.squeeze(0).numpy()
            #         rgb_view = torch.clamp(render_pkg["render"].detach().cpu(), 0.0, 1.0).permute(1, 2, 0).numpy()
            #         gt_view = gt_image.detach().cpu().permute(1, 2, 0).numpy()
            #
            #     all_point_clouds = []
            #     for i in range(img_index - 2, img_index + 2):
            #         # Lidar coord
            #         example_gaussian = gaussians.gaussian_model_list[i]
            #         # point_cloud = example_gaussian.original_point_cloud.detach()
            #         point_cloud = example_gaussian._initial_means.detach()
            #         # Convert to world
            #         rotation_wl, translation_wl = scene.sensor_trajectory.get_lidar_pose(i)
            #         rotation_wl = pytorch3d.transforms.quaternion_to_matrix(rotation_wl.detach())
            #         translation_wl = translation_wl.detach()
            #         point_cloud_world = point_cloud @ rotation_wl.T + translation_wl
            #         #
            #         # point_cloud_world = example_gaussian._xyz.detach()
            #         all_point_clouds.append(point_cloud_world)
            #
            #     points_world = torch.concatenate(all_point_clouds, dim=0)
            #     rotation_wl_curr, translation_wl_curr = scene.sensor_trajectory.get_lidar_pose(img_index)
            #     rotation_wl_curr = pytorch3d.transforms.quaternion_to_matrix(rotation_wl_curr.detach())
            #     translation_wl_curr = - rotation_wl_curr.T @ translation_wl_curr
            #     rotation_wl_curr = rotation_wl_curr.T
            #     points_lidar = points_world @ rotation_wl_curr.T + translation_wl_curr
            #     points_lidar = points_lidar.detach().cpu().numpy()
            #     rotation_cl, translation_cl = scene.sensor_trajectory.get_extrinsics()
            #     # rotation_cl_numpy = rotation_cl.detach().cpu().numpy()
            #     # translation_cl_numpy = translation_cl.detach().cpu().numpy()
            #     rotation_cl = pytorch3d.transforms.quaternion_to_matrix(rotation_cl.detach()).cpu().numpy()
            #     translation_cl = translation_cl.detach().cpu().numpy()
            #
            #     T_cl = np.eye(4)
            #     T_cl[:3, :3] = rotation_cl
            #     T_cl[:3, 3] = translation_cl
            #
            #     viewpoint_stack_debug = scene.getTrainCameras().copy()
            #     trained_images = scene.getTrainImages()
            #     original_img = trained_images[img_index]
            #     P = viewpoint_stack_debug[img_index].projection_matrix.detach().cpu().transpose(0, 1).numpy()
            #     projected_img2 = project_lidar_to_image_with_projection(points_lidar, original_img, P, T_cl)
            #
            #     if depth_view is not None:
            #         cv2.imshow('projected_img', projected_img2)
            #     if depth_view is not None:
            #         cv2.imshow('depth_view', depth_view)
            #     cv2.waitKey(3)


            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                rotation_err, translation_err = scene.sensor_trajectory.get_extrinsic_error()
                if torch.is_tensor(match_loss):
                    match_loss_info = match_loss.item()
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "rotation_err": rotation_err.numpy().tolist(),
                                          "translation_err": translation_err.numpy().tolist()})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()


            # Optimizer step
            if iteration < opt.iterations:
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                if iteration >= change_iteration[current_scale_index] + 1000:
                    scene.sensor_trajectory.start_optimize_rotation_cl()
                    scene.sensor_trajectory.start_optimize_translation_cl()

                    if iteration >= change_iteration[current_scale_index] + per_level_iterations[
                        current_scale_index] - 10:
                        scene.sensor_trajectory.stop_optimize_rotation_cl()
                        scene.sensor_trajectory.stop_optimize_translation_cl()


                    if iteration % 30 ==0 or (current_scale_index <= 2 and iteration %10 == 0):
                        scene.sensor_trajectory.optimizer.step()
                        scene.sensor_trajectory.update_extrinsics()
                        scene.sensor_trajectory.optimizer.zero_grad(set_to_none=True)
                        all_depths, all_depth_masks = generate_all_depths(scene, pipe, background,
                                                                          resolution_scales[current_scale_index],
                                                                          train_test_exp=dataset.train_test_exp,
                                                                          full_cut_size=cut_full_size,
                                                                          min_depth=min_depth, max_depth=max_depth)
                else:
                    scene.sensor_trajectory.stop_optimize_rotation_cl()
                    scene.sensor_trajectory.stop_optimize_translation_cl()

    rotation_err, translation_err = scene.sensor_trajectory.get_extrinsic_error()
    print('Final rotation error:', torch.norm(rotation_err).item())
    print('Final translation error:', torch.norm(translation_err).item())

    print('Calibrated rotation: ')
    print(pytorch3d.transforms.quaternion_to_matrix(scene.sensor_trajectory.rotation_cl.detach().cpu()).numpy())
    print('Calibrated translation: ')
    print(scene.sensor_trajectory.translation_cl.detach().cpu().numpy())


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[10, 1_000, 5_000, 7_000, 10000, 15000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 5_000, 7_000, 10000, 15000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[3000, 5000, 10000, 15000, 20000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--cam_id", type=str, default=None)
    parser.add_argument("--data_seq", type=int, default=None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    lp.cam_id = args.cam_id
    lp.data_seq = args.data_seq
    # print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)  # 开启异常检测
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
