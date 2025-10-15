import numpy as np
import cv2
import matplotlib.pyplot as plt
import threading
from queue import Queue, Empty
from multiprocessing import Lock, Process
import torch.nn.functional as F
import torch


def project_lidar_to_image(points_lidar, image, K, T_lidar_to_cam, P = None):
    """
    将雷达点云投影到图像平面并可视化
    参数：
        points_lidar: Nx3 numpy数组，雷达坐标系下的点云
        image: HxWx3 numpy数组，BGR格式的彩色图像
        K: 3x3 numpy数组，相机内参矩阵
        T_lidar_to_cam: 4x4 numpy数组，雷达到相机的变换矩阵
    返回：
        HxWx3 numpy数组，融合后的可视化图像
    """
    # 转换点云到齐次坐标
    N = points_lidar.shape[0]
    points_hom = np.hstack((points_lidar, np.ones((N, 1))))

    # 转换到相机坐标系
    points_cam = (T_lidar_to_cam @ points_hom.T).T[:, :3]

    # 过滤掉相机后方的点
    valid_z = points_cam[:, 2] > 0.1  # 0.1米阈值防止过近点
    points_cam = points_cam[valid_z]

    if points_cam.shape[0] == 0:
        return image.copy()

    # 提取相机参数
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 投影到像素平面
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    u = (fx * x / z + cx).astype(int)
    v = (fy * y / z + cy).astype(int)

    # 过滤有效像素坐标
    H, W = image.shape[:2]
    valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid_uv]
    v = v[valid_uv]
    z = z[valid_uv]

    if len(u) == 0:
        return image.copy()

    # 创建深度颜色映射
    z_min, z_max = np.percentile(z, [2, 98])  # 使用98%分位数避免异常值
    norm = plt.Normalize(z_min, z_max)
    cmap = plt.get_cmap('jet')  # 使用jet色图增强对比

    # 生成颜色并转换到0-255范围
    colors = cmap(norm(z))[:, :3]  # 忽略alpha通道
    colors = (colors * 255).astype(np.uint8)

    # 创建深度缓冲区
    depth_buffer = np.full((H, W), np.inf)
    result = image.copy()

    # 更新每个有效点
    for i in range(len(u)):
        if z[i] < depth_buffer[v[i], u[i]]:
            depth_buffer[v[i], u[i]] = z[i]
            result[v[i], u[i]] = colors[i]

    return result


def project_to_pixel_torch(points_3d, P, image_width, image_height):
    """将视图空间的3D点投影到图像像素坐标（NumPy版本）"""
    # points_3d: (N, 3) 视图空间坐标（需先应用视图变换）
    # 转换为齐次坐标 (N, 4)
    ones = torch.ones_like(points_3d[:,0:1])
    points_homo = torch.hstack([points_3d, ones])  # (N, 4)

    # 应用投影矩阵 → 裁剪空间 (N, 4)
    clip_coords = points_homo @ P.T  # 等价于 np.matmul(points_homo, P.T)

    # 透视除法 → NDC (范围根据坐标系调整)
    w = clip_coords[:, 3].view(-1, 1)  # (N, 1)
    ndc_coords = clip_coords[:, :3] / w    # (N, 3)

    # 视口变换 → 图像像素坐标 (u, v)
    u = (ndc_coords[:, 0] + 1) * 0.5 * image_width
    v = (ndc_coords[:, 1] + 1) * 0.5 * image_height  # 翻转y方向

    return torch.column_stack([u, v])  # 合并为 (N, 2)

def project_to_pixel(points_3d, P, image_width, image_height):
    """将视图空间的3D点投影到图像像素坐标（NumPy版本）"""
    # points_3d: (N, 3) 视图空间坐标（需先应用视图变换）
    # 转换为齐次坐标 (N, 4)
    ones = np.ones((points_3d.shape[0], 1), dtype=np.float32)
    points_homo = np.hstack([points_3d, ones])  # (N, 4)

    # 应用投影矩阵 → 裁剪空间 (N, 4)
    clip_coords = points_homo @ P.T  # 等价于 np.matmul(points_homo, P.T)

    # 透视除法 → NDC (范围根据坐标系调整)
    w = clip_coords[:, 3].reshape(-1, 1)  # (N, 1)
    ndc_coords = clip_coords[:, :3] / w    # (N, 3)

    # 视口变换 → 图像像素坐标 (u, v)
    u = (ndc_coords[:, 0] + 1) * 0.5 * image_width
    v = (ndc_coords[:, 1] + 1) * 0.5 * image_height  # 翻转y方向

    return np.column_stack([u, v])  # 合并为 (N, 2)


def project_lidar_to_image_with_projection(points_lidar, image, P, T_lidar_to_cam):
    """
    将雷达点云投影到图像平面并可视化
    参数：
        points_lidar: Nx3 numpy数组，雷达坐标系下的点云
        image: HxWx3 numpy数组，BGR格式的彩色图像
        K: 3x3 numpy数组，相机内参矩阵
        T_lidar_to_cam: 4x4 numpy数组，雷达到相机的变换矩阵
    返回：
        HxWx3 numpy数组，融合后的可视化图像
    """
    # 转换点云到齐次坐标
    N = points_lidar.shape[0]
    points_hom = np.hstack((points_lidar, np.ones((N, 1))))

    # 转换到相机坐标系
    points_cam = (T_lidar_to_cam @ points_hom.T).T[:, :3]

    # 过滤掉相机后方的点
    valid_z = points_cam[:, 2] > 0.1  # 0.1米阈值防止过近点
    points_cam = points_cam[valid_z]

    if points_cam.shape[0] == 0:
        return image.copy()

    # 提取相机参数

    # 投影到像素平面
    z = points_cam[:, 2]

    H, W = image.shape[:2]
    points_pixel = project_to_pixel(points_cam, P, W, H)

    u = points_pixel[:,0].astype(int)
    v = points_pixel[:,1].astype(int)

    # 过滤有效像素坐标
    valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid_uv]
    v = v[valid_uv]
    z = z[valid_uv]

    if len(u) == 0:
        return image.copy()

    # 创建深度颜色映射
    z_min, z_max = np.percentile(z, [2, 98])  # 使用98%分位数避免异常值
    norm = plt.Normalize(z_min, z_max)
    cmap = plt.get_cmap('jet')  # 使用jet色图增强对比

    # 生成颜色并转换到0-255范围
    colors = cmap(norm(z))[:, :3]  # 忽略alpha通道
    colors = (colors * 255).astype(np.uint8)

    # 创建深度缓冲区
    depth_buffer = np.full((H, W), np.inf)
    result = image.copy()

    # 更新每个有效点
    for i in range(len(u)):
        if z[i] < depth_buffer[v[i], u[i]]:
            depth_buffer[v[i], u[i]] = z[i]
            result[v[i], u[i]] = colors[i]

    return result




class ImageViewer:
    def __init__(self, img = None):
        self.queue = Queue(maxsize=1)  # 单缓冲队列
        self.running = False
        self.window_name = "OpenCV Viewer"
        self.img = img
        self.mutex = Lock()

    def initialize_img(self, lidar_points, img, intrinsics, T_cl):
        self.img = project_lidar_to_image(lidar_points, img, intrinsics, T_cl)
        self.lidar_points = lidar_points
        self.intrinsics = intrinsics
        self.T_cl = T_cl
        self.start()

    def update_pose(self, T_cl):
        self.T_cl = T_cl
        self.mutex.acquire()
        self.img = project_lidar_to_image(self.lidar_points, self.img, self.intrinsics, self.T_cl)
        self.mutex.release()

    def _display_thread(self):
        """显示线程内部函数"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        while self.running:
            self.mutex.acquire()
            # 非阻塞获取图像，最多等待100ms
            cv2.imshow(self.window_name, self.img)
            # 必须定期处理窗口事件
            key = cv2.waitKey(30) & 0xFF
            self.mutex.release()
            if key == 27:  # ESC键退出
                self.stop()
        cv2.destroyAllWindows()

    def start(self):
        """启动显示线程"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._display_thread)
            self.thread.daemon = True  # 设置为守护线程
            self.thread.start()

    def update(self, image):
        """更新显示图像（线程安全）"""
        try:
            self.mutex.acquire()
            self.img = image
            self.mutex.release()
        except:
            pass

    def stop(self):
        """停止显示线程"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1)



def unproject_pixels_to_3d_torch(pixel_uv, depth_values, P, image_width, image_height):
    """
    Unprojects 2D pixel coordinates with depth back to 3D points in view space.
    This is the inverse of project_to_pixel_torch, assuming depth_values are Z_view.

    Args:
        pixel_uv (torch.Tensor): (N, 2) tensor of (u, v) pixel coordinates.
        depth_values (torch.Tensor): (N,) or (N,1) tensor of depth values (Z_view) for each pixel.
                                     These are the Z coordinates in the view/camera space.
        P (torch.Tensor): (4, 4) projection matrix used in the forward projection.
        image_width (float): Width of the image in pixels.
        image_height (float): Height of the image in pixels.

    Returns:
        torch.Tensor: (N, 3) tensor of 3D points (X_view, Y_view, Z_view) in view space.
                      Returns empty tensor if input pixel_uv is empty.
    """
    if pixel_uv.shape[0] == 0:
        return torch.empty((0, 3), device=pixel_uv.device, dtype=pixel_uv.dtype)

    u = pixel_uv[:, 0]
    v = pixel_uv[:, 1]
    Zv = depth_values.squeeze(-1) if depth_values.ndim == 2 else depth_values  # Ensure (N,)

    # Inverse viewport transform: Pixel to NDC
    # u = (x_ndc + 1) * 0.5 * image_width  => x_ndc = (2u / image_width) - 1
    # v = (y_ndc + 1) * 0.5 * image_height => y_ndc = (2v / image_height) - 1
    # This matches the forward function's NDC to pixel mapping.
    x_ndc = (2.0 * u / image_width) - 1.0
    y_ndc = (2.0 * v / image_height) - 1.0

    # We need to solve for Xv, Yv. We know Zv.
    # The forward projection equations are:
    # x_ndc * Wc = P[0,0]Xv + P[0,1]Yv + P[0,2]Zv + P[0,3]  (1)
    # y_ndc * Wc = P[1,0]Xv + P[1,1]Yv + P[1,2]Zv + P[1,3]  (2)
    # Wc         = P[3,0]Xv + P[3,1]Yv + P[3,2]Zv + P[3,3]  (3)
    #
    # This is a system of 3 linear equations for Xv, Yv, Wc for each point:
    # P[0,0]Xv + P[0,1]Yv - x_ndc*Wc = - (P[0,2]Zv + P[0,3])
    # P[1,0]Xv + P[1,1]Yv - y_ndc*Wc = - (P[1,2]Zv + P[1,3])
    # P[3,0]Xv + P[3,1]Yv - Wc       = - (P[3,2]Zv + P[3,3])
    #
    # Let sol = [Xv, Yv, Wc]^T.
    # M @ sol = B
    # M is (N, 3, 3), B is (N, 3, 1)

    num_points = pixel_uv.shape[0]
    M_batch = torch.zeros((num_points, 3, 3), device=P.device, dtype=P.dtype)
    B_batch = torch.zeros((num_points, 3, 1), device=P.device, dtype=P.dtype)

    # Construct M matrix for each point
    M_batch[:, 0, 0] = P[0, 0]
    M_batch[:, 0, 1] = P[0, 1]
    M_batch[:, 0, 2] = -x_ndc

    M_batch[:, 1, 0] = P[1, 0]
    M_batch[:, 1, 1] = P[1, 1]
    M_batch[:, 1, 2] = -y_ndc

    M_batch[:, 2, 0] = P[3, 0]
    M_batch[:, 2, 1] = P[3, 1]
    M_batch[:, 2, 2] = -1.0

    # Construct B vector for each point
    B_batch[:, 0, 0] = -(P[0, 2] * Zv + P[0, 3])
    B_batch[:, 1, 0] = -(P[1, 2] * Zv + P[1, 3])
    B_batch[:, 2, 0] = -(P[3, 2] * Zv + P[3, 3])

    # Solve the batched linear system
    # Some matrices in M_batch might be singular if P or ndc coords are weird.
    # e.g. if P[3,0]=P[3,1]=0 and -1=0 (problematic) or if leading 2x2 of P is singular.
    # Using try-catch or checking determinant could be added for robustness.
    try:
        solution_batch = torch.linalg.solve(M_batch, B_batch)  # Result is (N, 3, 1)
    except torch.linalg.LinAlgError as e:
        # Handle cases where matrix is singular for some points
        # For now, re-raise or return NaNs.
        # A more robust solution might involve SVD or pseudo-inverse,
        # or identifying problematic points.
        print(f"Linear algebra error during unprojection: {e}")
        # Create a NaN tensor of the expected shape
        nan_points_3d = torch.full((num_points, 3), float('nan'), device=P.device, dtype=P.dtype)
        return nan_points_3d

    Xv_batch = solution_batch[:, 0, 0]
    Yv_batch = solution_batch[:, 1, 0]
    # Wc_batch = solution_batch[:, 2, 0] # Wc is also solved for, can be used for sanity checks

    # The 3D point is (Xv, Yv, Zv)
    points_3d_unprojected = torch.stack([Xv_batch, Yv_batch, Zv], dim=-1)

    return points_3d_unprojected


# ----- Helper function to use the unprojection with a mask and depth_map -----
def get_3d_points_from_pixels_depth_mask_torch(
        depth_map,
        mask,
        P,
        image_width,
        image_height
):
    """
    Extracts 3D points from a depth map using a mask and unprojects them.

    Args:
        depth_map (torch.Tensor): (H, W) tensor of depth values (Z_view).
        mask (torch.Tensor): (H, W) boolean tensor, True for pixels to unproject.
        P (torch.Tensor): (4, 4) projection matrix.
        image_width (float): Width of the image.
        image_height (float): Height of the image.

    Returns:
        torch.Tensor: (N_masked, 3) tensor of 3D points in view space,
                      where N_masked is the number of True values in the mask.
                      Returns empty tensor if no pixels in mask.
    """
    if not torch.any(mask):
        return torch.empty((0, 3), device=depth_map.device, dtype=depth_map.dtype)

    # Get pixel coordinates (v, u) where mask is True
    v_coords, u_coords = torch.where(mask)  # v_coords are row indices, u_coords are col indices

    # Select depth values for these pixels
    depth_values_masked = depth_map[v_coords, u_coords]  # (N_masked,)

    # Stack u, v coordinates
    # Note: u_coords correspond to x-direction (width), v_coords to y-direction (height)
    pixel_uv_masked = torch.stack([u_coords.float(), v_coords.float()], dim=-1)  # (N_masked, 2)

    # Unproject
    points_3d_view_space = unproject_pixels_to_3d_torch(
        pixel_uv_masked,
        depth_values_masked,  # Will be (N_masked,)
        P,
        image_width,
        image_height
    )
    return points_3d_view_space


# 示例用法
if __name__ == "__main__":
    # 生成虚拟数据
    H, W = 480, 640
    dummy_image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    dummy_image = dummy_image * 0 + 122

    # 虚拟雷达点云（10米范围内的立方体）
    x = np.random.uniform(-5, 5, 10000)
    y = np.random.uniform(-5, 5, 10000)
    z = np.random.uniform(2, 10, 10000)
    points_lidar = np.vstack([x, y, z]).T

    # 虚拟相机参数
    K = np.array([[500, 0, 320],
                  [0, 500, 240],
                  [0, 0, 1]])

    # 虚拟外参（相机位于雷达前方1米处）
    T_lidar_to_cam = np.eye(4)
    T_lidar_to_cam[2, 3] = 1.0

    # 执行投影
    vis_image = project_lidar_to_image(points_lidar, dummy_image, K, T_lidar_to_cam)

    # 显示结果
    cv2.imshow('Projection Result', vis_image[..., ::-1])  # 转换BGR到RGB
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_gradient_mask(image_tensor, ratio = 0.5):
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
    gx = F.conv2d(image_tensor.unsqueeze(0), sobel_kernel_x, padding=1, groups=image_tensor.shape[0]).squeeze(0)
    gy = F.conv2d(image_tensor.unsqueeze(0), sobel_kernel_y, padding=1, groups=image_tensor.shape[0]).squeeze(0)

    # 计算梯度幅值（每个通道独立计算，然后取平均）：sqrt(gx^2 + gy^2)
    gradient_magnitude = torch.sqrt(gx.pow(2) + gy.pow(2)).mean(dim=0)  # 形状 [H, W]

    # 步骤2: 计算图像的平均值（参考网页6的均值计算逻辑[6](@ref)）
    image_mean = torch.mean(image_tensor)  # 整个图像的平均值（标量）

    # 步骤3: 生成 mask（梯度幅值大于平均值的像素为 1，否则为 0），参考网页9的 mask 生成方法[9](@ref)
    mask = (gradient_magnitude > image_mean * ratio).float()  # 二值化，形状 [H, W]

    return mask