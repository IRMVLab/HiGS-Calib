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

import torch
import sys
from datetime import datetime
import numpy as np
import random
import cv2
import torch.nn.functional as F

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution, image_cutt = 0):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    resized_image = resized_image[image_cutt:]
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def PILtoTorchUndistorted(pil_image, resolution,K, D, image_cutt = 0):
    resized_image_PIL = pil_image.resize(resolution)
    image_array = np.array(resized_image_PIL)
    image_array = cv2.undistort(image_array, K, D)
    resized_image = torch.from_numpy(image_array) / 255.0
    resized_image = resized_image[image_cutt:]
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def sample_uniform(min_tensor, max_tensor, M):
    """
    在两个给定的张量定义的边界内进行均匀随机采样

    参数:
    min_tensor (Tensor): 包含每个维度最小值的N维张量
    max_tensor (Tensor): 包含每个维度最大值的N维张量
    M (int): 采样数量

    返回:
    Tensor: 形状为(M, N)的张量，包含随机采样的点
    """
    # 验证输入
    if min_tensor.dim() != max_tensor.dim():
        raise ValueError("min_tensor 和 max_tensor 必须具有相同的维度")
    if min_tensor.dim() != 1:
        raise ValueError("输入张量应为1D（N维向量）")
    if not torch.all(min_tensor <= max_tensor):
        raise ValueError("所有 min_tensor 值必须小于等于 max_tensor 值")
    if M <= 0:
        raise ValueError("采样数量 M 必须是正整数")

    # 获取维度数量
    N = min_tensor.size(0)

    # 生成均匀分布的随机数 [0, 1)
    rand_samples = torch.rand(M, N).to(min_tensor.device)

    # 缩放到指定范围
    scaled_samples = min_tensor + rand_samples * (max_tensor - min_tensor)

    return scaled_samples

def create_gradient_mask(image_tensor, ratio = 0.1):
    """
    基于 PyTorch 检查图像的彩色像素梯度是否大于图像平均值，并生成二值 mask。

    参数:
        image_tensor (torch.Tensor): 输入图像张量，形状为 [C, H, W]（C为通道数，H为高度，W为宽度）。
                                    建议图像值范围 [0, 1] 或已标准化。

    返回:
        torch.Tensor: 二值 mask，形状为 [H, W]。值为 1 表示对应像素的梯度大于图像平均值，否则为 0。
    """
    # 确保输入是浮点型张量，避免类型错误
    if image_tensor.dtype != torch.float32:
        image_tensor = image_tensor.float()

    # 步骤1: 计算图像的梯度幅值（使用 Sobel 算子）
    # 定义 Sobel 卷积核（水平和垂直方向），参考网页4的卷积核设计[4](@ref)
    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image_tensor.device)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image_tensor.device)

    # 扩展卷积核以匹配输入图像的通道数（适用于彩色图像）
    sobel_kernel_x = sobel_kernel_x.repeat(image_tensor.shape[0], 1, 1, 1)  # 形状 [C, 1, 3, 3]
    sobel_kernel_y = sobel_kernel_y.repeat(image_tensor.shape[0], 1, 1, 1)

    # 应用卷积计算梯度的水平和垂直分量
    pad_width = (1, 1, 1, 1)
    image_pad = F.pad(
        image_tensor.unsqueeze(0),
        pad_width,
        mode='replicate'  # 可选：'constant'（默认0）、'reflect'、'replicate'、'circular'
    )
    gx = F.conv2d(image_pad, sobel_kernel_x, padding=0, groups=image_pad.shape[1]).squeeze(0)
    gy = F.conv2d(image_pad, sobel_kernel_y, padding=0, groups=image_pad.shape[1]).squeeze(0)

    # 计算梯度幅值（每个通道独立计算，然后取平均）：sqrt(gx^2 + gy^2)
    gradient_magnitude = torch.sqrt(gx.pow(2) + gy.pow(2)).mean(dim=0)  # 形状 [H, W]

    # 步骤2: 计算梯度的平均值
    grad_mean = torch.mean(gradient_magnitude)  # 整个图像的平均值（标量）

    # 步骤3: 生成 mask（梯度幅值大于平均值的像素为 1，否则为 0），参考网页9的 mask 生成方法[9](@ref)
    mask = (gradient_magnitude > grad_mean * ratio).float()  # 二值化，形状 [H, W]


    c, h, w = image_tensor.shape
    h_grid_size = 3  # 38x38网格
    w_grid_size = int(w/h * h_grid_size)

    # 自适应块划分 [7,8](@ref)
    h_step = int(h / h_grid_size)
    w_step = int(w / w_grid_size)
    block_means = torch.zeros(h_grid_size, w_grid_size, device=image_tensor.device)

    # 5. 遍历每个块计算平均梯度
    for i in range(h_grid_size):
        for j in range(w_grid_size):
            # 计算块边界 [7](@ref)
            h_start = int(i * h_step)
            h_end = min(int((i + 1) * h_step), h)
            w_start = int(j * w_step)
            w_end = min(int((j + 1) * w_step), w)

            # 提取当前块区域
            block = gradient_magnitude[h_start:h_end, w_start:w_end]

            # 计算块平均梯度 [1](@ref)
            block_means[i, j] = block.mean()

    # 标记低梯度块 [13](@ref)
    threshold = grad_mean * ratio
    for i in range(h_grid_size):
        for j in range(w_grid_size):
            h_start = int(i * h_step)
            h_end = min(int((i + 1) * h_step), h)
            w_start = int(j * w_step)
            w_end = min(int((j + 1) * w_step), w)

            if block_means[i, j] < threshold:
                mask[h_start:h_end, w_start:w_end] = 0.

    return gradient_magnitude/torch.max(gradient_magnitude), mask


