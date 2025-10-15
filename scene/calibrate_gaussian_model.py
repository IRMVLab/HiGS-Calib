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
import pytorch3d.transforms
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.voxels import *

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass


class CalibrateGaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)  # 构造协方差矩阵Σ=RS(SR^T)
            symm = strip_symmetric(actual_covariance)  # 因为是对陈矩阵，所以只需要存储六个元素
            return symm

        self.scaling_activation = torch.exp  # 缩放参数实际值: exp(log_scaling)
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid  # 透明度: sigmoid(log_opacity)
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize  # 四元数单位化

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree
        self._rotation_wl = None
        self._translation_wl = None
        self._projection_matrix = None
        self.original_point_cloud = None
        self._initial_means = None

        self.spatial_lr_scale = 0
        self.setup_functions()

    #TODO: Not implemented yet.
    # def capture(self):
    #     return (
    #         self.active_sh_degree,
    #         self._rotation_wl,
    #         self._translation_wl,
    #     )


    #TODO: Not implemented yet.
    def restore(self, model_args, training_args):
        # (self.active_sh_degree,
        #  self._xyz,
        #  self._features_dc,
        #  self._features_rest,
        #  self._scaling,
        #  self._rotation,
        #  self._opacity,
        #  self.max_radii2D,
        #  xyz_gradient_accum,
        #  denom,
        #  opt_dict,
        #  self.spatial_lr_scale) = model_args
        # self.training_setup(training_args)
        # self.xyz_gradient_accum = xyz_gradient_accum
        # self.denom = denom
        # self.optimizer.load_state_dict(opt_dict)
        return


    # 使用点云进行模型初始化
    def create_from_pcd(self, pcd: BasicPointCloud,
                        spatial_lr_scale: float,
                        rotation_wl_param: nn.Parameter,
                        translation_wl_param: nn.Parameter):
        #rotation_wl 3*3 matrix
        #translation_wl 3*1 vector
        self._rotation_wl = rotation_wl_param
        self._translation_wl = translation_wl_param
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        self._initial_means = fused_point_cloud
        self.original_point_cloud = fused_point_cloud

class CalibrateGaussianListModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)  # 构造协方差矩阵Σ=RS(SR^T)
            symm = strip_symmetric(actual_covariance)  # 因为是对陈矩阵，所以只需要存储六个元素
            return symm

        self.scaling_activation = torch.exp  # 缩放参数实际值: exp(log_scaling)
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid  # 透明度: sigmoid(log_opacity)
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize  # 四元数单位化

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree

        self.gaussian_model_list = []
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)  # 高斯中心坐标 [N,3]
        self._features_dc = torch.empty(0)  # 球谐函数DC分量 [N,1,3] 0阶球谐分量（漫反射颜色)
        self._features_rest = torch.empty(0)  # 球谐高阶分量 [N, (sh_degree+1)^2-1, 3] 1阶及更高阶分量（镜面反射等高频细节）
        self._scaling = torch.empty(0)  # 缩放对数形式 [N,3]
        self._rotation = torch.empty(0)  # 四元数 [N,4]
        self._opacity = torch.empty(0)  # 不透明度对数形式 [N,1]
        self.max_radii2D = torch.empty(0)  # 各高斯在2D投影的最大半径 [N]
        self.xyz_gradient_accum = torch.empty(0)  # 位置梯度累积量 [N,1]
        self.denom = torch.empty(0)  # 梯度归一化分母 [N,1]


        self.color_encoder = None
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # 参数激活函数
        self.setup_functions()

    def capture(self):
        capture_list = {}
        for index, model in enumerate(self.gaussian_model_list):
            capture_list[f'model_{index}'] = model.capture()
        return capture_list

    #TODO: Not implemented yet
    def restore(self, model_args, training_args):
        # (self.active_sh_degree,
        #  self._xyz,
        #  self._features_dc,
        #  self._features_rest,
        #  self._scaling,
        #  self._rotation,
        #  self._opacity,
        #  self.max_radii2D,
        #  xyz_gradient_accum,
        #  denom,
        #  opt_dict,
        #  self.spatial_lr_scale) = model_args
        # self.training_setup(training_args)
        # self.xyz_gradient_accum = xyz_gradient_accum
        # self.denom = denom
        # self.optimizer.load_state_dict(opt_dict)
        return

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    # 使用点云进行模型初始化
    def create_from_pcd_list(self, pcd_list: list, spatial_lr_scale: float, rotations_wl_list: list, translations_wl_list: list,  projection_matrix, resolution:float):
        assert len(pcd_list) == len(rotations_wl_list)
        assert len(pcd_list) == len(translations_wl_list)
        #Initialize gaussian model one by one
        points_list = []
        colors_list = []
        gaussian_num = len(pcd_list)
        for i in range(gaussian_num):
            points = torch.Tensor(np.asarray(pcd_list[i].points)).float().cuda()
            rotation_wl = pytorch3d.transforms.quaternion_to_matrix(rotations_wl_list[i].detach())
            translation_wl = translations_wl_list[i].detach()
            points = points @ rotation_wl.T + translation_wl
            points_list.append(points)
            colors_list.append(torch.Tensor(pcd_list[i].colors).float().cuda())
        all_points = torch.concatenate(points_list, dim=0)
        all_colors = torch.concatenate(colors_list, dim=0)
        self.create_from_pcd(fused_point_cloud = all_points, colors= all_colors, spatial_lr_scale=spatial_lr_scale, resolution = resolution)
        for i in range(gaussian_num):
            self.gaussian_model_list.append(CalibrateGaussianModel(self.max_sh_degree, self.optimizer_type))
        self.spatial_lr_scale = spatial_lr_scale
        for i in range(len(pcd_list)):
            self.gaussian_model_list[i]._projection_matrix = projection_matrix
            self.gaussian_model_list[i].create_from_pcd(pcd_list[i], spatial_lr_scale, rotations_wl_list[i], translations_wl_list[i])




    # 使用点云进行模型初始化
    def create_from_pcd(self, fused_point_cloud, colors,
                        spatial_lr_scale: float,
                        resolution: float):

        self.spatial_lr_scale = spatial_lr_scale
        self.original_point_cloud = fused_point_cloud

        final_means_large, final_covariances_large = fit_gaussians_in_voxels(
            points=fused_point_cloud,
            voxel_size=resolution,
            default_cov_diag_value=resolution * resolution * 0.2
        )
        #
        fused_point_cloud = final_means_large

        # min_bound = torch.min(fused_point_cloud, dim=0)
        # max_bound = torch.max(fused_point_cloud, dim=0)
        # self.min_bound = min_bound
        # self.max_bound = max_bound
        # # aabb = 2
        # # hparams = {
        # #     'aabb': aabb,
        # #     'chunk_size': 1024*16
        # # }
        # # self.color_encoder = ColorEncoder


        fused_color = RGB2SH(torch.ones_like(fused_point_cloud) * colors[0, 0])

        # TODO: Culling points that are not visible in the lidar
        # self.culling_points()

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # 高阶球谐设置为0
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        scales = torch.log(torch.diagonal(torch.sqrt(final_covariances_large), dim1=1, dim2=2))
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")  # 初始旋转（无旋转）
        rots[:, 0] = 1

        # Change opacities to 0.99
        opacities = self.inverse_opacity_activation(
            0.99 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))  # 初始透明度

        self._xyz = nn.Parameter(fused_point_cloud).requires_grad_(True)
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous()).requires_grad_(False)
        self._scaling = nn.Parameter(scales).requires_grad_(True)
        self._rotation = nn.Parameter(rots).requires_grad_(True)
        self._opacity = nn.Parameter(opacities).requires_grad_(True)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def densify_gaussian_model(self, resolution: float):
        fused_point_cloud = self.original_point_cloud
        #Offer a small delta to the position
        stds = self.get_scaling
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)

        cloned_XYZ = self._xyz.clone() + samples
        self._xyz = nn.Parameter(torch.cat([self._xyz, cloned_XYZ], dim=0)).requires_grad_(True)
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, self._features_dc.clone()], dim=0)).requires_grad_(True)
        self._features_rest = nn.Parameter(
            torch.cat([self._features_rest, self._features_rest.clone()], dim=0)).requires_grad_(True)
        self._scaling = nn.Parameter(
            torch.cat([self._scaling, self._scaling.clone()], dim=0)).requires_grad_(True)
        self._rotation = nn.Parameter(
            torch.cat([self._rotation, self._rotation.clone()], dim=0)).requires_grad_(True)
        self._opacity = nn.Parameter(
            torch.cat([self._opacity, self._opacity.clone()], dim=0)).requires_grad_(True)
        self.training_setup(self.training_args)

    def stop_optimize(self):
        attribute_list = [self._xyz, self._features_dc, self._features_rest, self._scaling, self._rotation, self._opacity]
        for attribute in attribute_list:
            attribute.requires_grad_(False)
            attribute.grad = None

    def start_to_optimize(self):
        attribute_list = [self._xyz, self._features_dc, self._features_rest, self._scaling, self._rotation,
                          self._opacity]
        for attribute in attribute_list:
            attribute.requires_grad_(True)

    def need_to_optimize(self):
        attribute_list = [self._xyz, self._features_dc, self._features_rest, self._scaling, self._rotation,
                          self._opacity]
        for attribute in attribute_list:
            if attribute.requires_grad == True:
                return True
        return False

    def training_setup(self, training_args):
        self.training_args = training_args
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # SparseGaussianAdam仅更新可见高斯参数，提升效率
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.get_features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.get_features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.get_opacity.detach().cpu().numpy()
        scale = self.get_scaling.detach().cpu().numpy()
        rotation = self.get_rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    #TODO: continue to modify the state in the global optimizer
    def reset_opacity(self):
        optimizable_tensor_list = []
        for model in self.gaussian_model_list:
            optimizable_tensor_list.append(model.reset_opacity())

        global_tensor_list = []
        for group in self.optimizer.param_groups:
            if group["name"] == 'opacity':
                for param_ind in range(len(optimizable_tensor_list)):
                    local_opacities = optimizable_tensor_list[param_ind]
                    stored_state = self.optimizer.state.get(group['params'][param_ind], None)
                    stored_state["exp_avg"] = torch.zeros_like(local_opacities)
                    stored_state["exp_avg_sq"] = torch.zeros_like(local_opacities)
                    del self.optimizer.state[group['params'][param_ind]]
                    group["params"][param_ind] = local_opacities['opacity']
                    self.optimizer.state[group['params'][param_ind]] = stored_state
                    global_tensor_list.append(group["params"][param_ind])
        optimizable_tensors = {}
        optimizable_tensors['opacity'] = global_tensor_list
        return optimizable_tensors


    def load_ply(self, path_list, use_train_test_exp=False):
        assert len(path_list) == len(self.gaussian_model_list)
        for index, model in enumerate(self.gaussian_model_list):
            model.load_ply(path_list[index], use_train_test_exp=use_train_test_exp)


