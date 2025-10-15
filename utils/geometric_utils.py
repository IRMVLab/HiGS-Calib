import torch

def find_nearest_gaussian(points, gaussians):
    """
    计算点云中每个点最近的高斯球索引

    Args:
        points (torch.Tensor): 输入点云，形状为[N, 3]
        gaussians (torch.Tensor): 高斯球中心坐标，形状为[M, 3]

    Returns:
        torch.Tensor: 每个点对应的最近高斯球索引，形状为[N]
    """
    # 计算点与高斯球之间的成对欧氏距离
    distances = torch.cdist(points, gaussians)  # 输出形状[N, M]

    # 获取最小距离的索引
    nearest_indices = torch.argmin(distances, dim=1)

    return nearest_indices

def batch_find_nearest(points, gaussians, batch_size=10000):
    nearest = []
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        dists = torch.cdist(batch, gaussians)
        nearest.append(torch.argmin(dists, dim=1))
    return torch.cat(nearest)