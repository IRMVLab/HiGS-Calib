import torch
import numpy as np  # Only for example data generation
import time  # For timing
import open3d as o3d

def compute_mean_covariance(points_in_voxel):

    num_points = points_in_voxel.shape[0]

    mean = torch.mean(points_in_voxel, dim=0)

    if num_points < 2:
        covariance = torch.ones((points_in_voxel.shape[1], points_in_voxel.shape[1]),
                                 device=points_in_voxel.device, dtype=points_in_voxel.dtype) * 1e-4
    else:
        centered_points = points_in_voxel - mean
        covariance = torch.matmul(centered_points.T, centered_points) / (num_points - 1)

    return mean, covariance


def adaptive_voxel_analysis_optimized(points_tensor, initial_voxel_size_val, threshold_val,
                                      min_points_per_voxel=4, max_depth=8, min_voxel_size_factor=0.05):

    if not torch.is_tensor(points_tensor):
        raise TypeError("Input points must be a PyTorch tensor.")
    if points_tensor.ndim != 2 or points_tensor.shape[1] != 3:
        raise ValueError("Input points_tensor must be a N x 3 tensor.")

    device = points_tensor.device
    dtype = points_tensor.dtype
    initial_voxel_size = torch.tensor(initial_voxel_size_val, device=device, dtype=dtype)
    threshold = torch.tensor(threshold_val, device=device, dtype=dtype)

    final_means = []
    final_covariances = []

    points_min = points_tensor.min(dim=0).values

    initial_voxel_indices_shifted = ((points_tensor - points_min) / initial_voxel_size).floor().to(torch.int32)

    unique_initial_voxels, inverse_indices = torch.unique(initial_voxel_indices_shifted, dim=0, return_inverse=True)

    processing_queue = []

    for i in range(unique_initial_voxels.shape[0]):
        original_indices_for_this_voxel = torch.where(inverse_indices == i)[0].to(torch.int32)

        if original_indices_for_this_voxel.shape[0] < min_points_per_voxel:
            if original_indices_for_this_voxel.shape[0] > 0:
                pts_in_vox = points_tensor[original_indices_for_this_voxel]
                mean, cov = compute_mean_covariance(pts_in_vox)
                if mean is not None:
                    final_means.append(mean)
                    final_covariances.append(cov)
            continue

        voxel_idx_coords = unique_initial_voxels[i].to(dtype)  # Ensure float for calculations
        current_voxel_min_abs = points_min + voxel_idx_coords * initial_voxel_size

        processing_queue.append({
            "point_indices": original_indices_for_this_voxel,
            "voxel_min_abs": current_voxel_min_abs,
            "voxel_size_scalar": initial_voxel_size_val,  # Use Python float for size tracking
            "depth": 0
        })

    del initial_voxel_indices_shifted, unique_initial_voxels, inverse_indices  # Free up memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    min_allowed_voxel_size_scalar = initial_voxel_size_val * min_voxel_size_factor

    while processing_queue:
        task = processing_queue.pop(0)

        current_point_indices = task["point_indices"]  # These are global indices
        current_voxel_min_abs = task["voxel_min_abs"]
        current_voxel_size_scalar = task["voxel_size_scalar"]
        current_depth = task["depth"]

        points_in_current_voxel = points_tensor[current_point_indices]

        if points_in_current_voxel.shape[0] < min_points_per_voxel:
            if points_in_current_voxel.shape[0] > 0:
                mean, covariance = compute_mean_covariance(points_in_current_voxel)
                if mean is not None:  # mean could be None if points_in_current_voxel becomes empty
                    final_means.append(mean)
                    final_covariances.append(covariance)
            continue

        mean, covariance = compute_mean_covariance(points_in_current_voxel)
        # covariance can be None if points_in_current_voxel has < 1 point, handled by check above
        # covariance can be all zeros if points_in_current_voxel has 1 point

        try:
            covariance_sym = (covariance + covariance.T) / 2.0  # Ensure symmetry for eigvalsh
            eigenvalues = torch.linalg.eigvalsh(covariance_sym)  # Ascending order
        except torch.linalg.LinAlgError:  # e.g. singular matrix from collinear points
            final_means.append(mean)
            final_covariances.append(covariance)
            continue

        if eigenvalues.shape[0] < 3:  # Not enough non-zero eigenvalues (e.g. points are co-planar/co-linear)
            final_means.append(mean)
            final_covariances.append(covariance)
            continue

        mu1 = eigenvalues[0]  # Smallest
        mu2 = eigenvalues[1]
        mu3 = eigenvalues[2]  # Largest

        norm_sq_eigenvalues = mu1 * mu1 + mu2 * mu2 + mu3 * mu3
        if norm_sq_eigenvalues < 1e-12:  # Avoid division by zero if all eigenvalues are tiny
            ratio = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            ratio = mu1 / torch.sqrt(norm_sq_eigenvalues)

        if torch.isnan(ratio): ratio = torch.tensor(0.0, device=device, dtype=dtype)

        subdivide = (ratio > threshold and
                     current_depth < max_depth and
                     (current_voxel_size_scalar / 2.0) >= min_allowed_voxel_size_scalar)

        if subdivide:
            new_voxel_size_scalar = current_voxel_size_scalar / 2.0
            # new_voxel_size_tensor = torch.tensor(new_voxel_size_scalar, device=device, dtype=dtype)

            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # offset_ijk = torch.tensor([i, j, k], device=device, dtype=dtype) * new_voxel_size_tensor
                        # More direct way for offset without creating small tensor repeatedly:
                        offset_vals = torch.tensor([i * new_voxel_size_scalar,
                                                    j * new_voxel_size_scalar,
                                                    k * new_voxel_size_scalar], device=device, dtype=dtype)

                        sub_voxel_min_abs = current_voxel_min_abs + offset_vals
                        sub_voxel_max_abs = sub_voxel_min_abs + new_voxel_size_scalar  # Max is exclusive

                        sub_mask_relative = (points_in_current_voxel >= sub_voxel_min_abs).all(dim=1) & \
                                            (points_in_current_voxel < sub_voxel_max_abs).all(dim=1)

                        indices_relative_to_current_voxel_batch = torch.where(sub_mask_relative)[
                            0]  # .to(torch.int32) if needed, but usually fine as is for indexing

                        if indices_relative_to_current_voxel_batch.shape[0] >= min_points_per_voxel:
                            new_global_indices_for_sub_voxel = current_point_indices[
                                indices_relative_to_current_voxel_batch]

                            processing_queue.append({
                                "point_indices": new_global_indices_for_sub_voxel,
                                "voxel_min_abs": sub_voxel_min_abs,
                                "voxel_size_scalar": new_voxel_size_scalar,
                                "depth": current_depth + 1
                            })
                        elif indices_relative_to_current_voxel_batch.shape[
                            0] > 0:  # Fewer than min_points but still some points
                            # Don't subdivide further, just record stats for this small sub-voxel
                            pts_in_sub_vox = points_in_current_voxel[indices_relative_to_current_voxel_batch]
                            sub_mean, sub_cov = compute_mean_covariance(pts_in_sub_vox)
                            if sub_mean is not None:
                                final_means.append(sub_mean)
                                final_covariances.append(sub_cov)
            del points_in_current_voxel, current_point_indices  # Help GC for these larger tensors
            if device.type == 'cuda':
                torch.cuda.empty_cache()  # Can be aggressive, use if memory is still tight

        else:  
            final_means.append(mean)
            final_covariances.append(covariance)
            del points_in_current_voxel, current_point_indices  # Help GC

    return final_means, final_covariances

def create_colored_point_cloud(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd


def fit_gaussians_in_voxels(points: torch.Tensor,
                            voxel_size: float or torch.Tensor,
                            default_cov_diag_value: float = 0.01,
                            min_bound: torch.Tensor = None,
                            max_bound: torch.Tensor = None):
    
    if points.shape[0] == 0:
        return torch.empty((0, 3), device=points.device, dtype=points.dtype), \
            torch.empty((0, 3, 3), device=points.device, dtype=points.dtype)

    device = points.device
    dtype = points.dtype

    if isinstance(voxel_size, (float, int)):
        voxel_size = torch.tensor([voxel_size, voxel_size, voxel_size], device=device, dtype=dtype)
    elif isinstance(voxel_size, torch.Tensor):
        if voxel_size.shape != (3,):
            raise ValueError("voxel_size张量的形状必须是 (3,)")
        voxel_size = voxel_size.to(device=device, dtype=dtype)
    else:
        raise TypeError("voxel_size必须是float或torch.Tensor类型")

    if voxel_size.min() <= 0:
        raise ValueError("voxel_size的所有元素必须大于0")

    if min_bound is None:
        data_min_coords = points.min(dim=0).values
        voxel_grid_origin = data_min_coords
    else:
        min_bound = min_bound.to(device=device, dtype=dtype)
        if min_bound.shape != (3,):
            raise ValueError("min_bound张量的形状必须是 (3,)")
        voxel_grid_origin = min_bound

    if max_bound is None:
        data_max_coords = points.max(dim=0).values
        grid_max_coord_for_dim_calc = data_max_coords
    else:
        max_bound = max_bound.to(device=device, dtype=dtype)
        if max_bound.shape != (3,):
            raise ValueError("max_bound张量的形状必须是 (3,)")
        grid_max_coord_for_dim_calc = max_bound
        
    voxel_indices = torch.floor((points - voxel_grid_origin) / voxel_size).long()

    grid_dims = torch.maximum(
        torch.floor((grid_max_coord_for_dim_calc - voxel_grid_origin) / voxel_size).long() + 1,
        torch.tensor([1, 1, 1], device=device, dtype=torch.long)  # 确保至少为1x1x1
    )

    voxel_indices = torch.max(voxel_indices, torch.zeros_like(voxel_indices))
    voxel_indices = torch.min(voxel_indices, grid_dims - 1)

    linear_voxel_indices = voxel_indices[:, 0] * grid_dims[1] * grid_dims[2] + \
                           voxel_indices[:, 1] * grid_dims[2] + \
                           voxel_indices[:, 2]

    unique_linear_indices, inverse_indices, counts = torch.unique(
        linear_voxel_indices, return_inverse=True, return_counts=True
    )
    num_non_empty_voxels = unique_linear_indices.shape[0]

    if num_non_empty_voxels == 0:
        return torch.empty((0, 3), device=device, dtype=dtype), \
            torch.empty((0, 3, 3), device=device, dtype=dtype)

    sum_points_per_voxel = torch.zeros((num_non_empty_voxels, 3), device=device, dtype=dtype)
    sum_points_per_voxel.index_add_(0, inverse_indices, points)
    voxel_means = sum_points_per_voxel / counts.unsqueeze(1)

    voxel_covariances = torch.zeros((num_non_empty_voxels, 3, 3), device=device, dtype=dtype)

    single_point_mask = (counts == 1)
    if single_point_mask.any():
        default_cov_matrix = torch.eye(3, device=device, dtype=dtype) * default_cov_diag_value
        voxel_covariances[single_point_mask] = default_cov_matrix  # 自动广播

    multi_point_mask = (counts > 1)
    if multi_point_mask.any():
        points_centered = points - voxel_means[inverse_indices]  # (N, 3)

        outer_products_centered = points_centered.unsqueeze(2) * points_centered.unsqueeze(1)

        sum_of_outer_products_centered = torch.zeros((num_non_empty_voxels, 3, 3), device=device, dtype=dtype)
        sum_of_outer_products_centered.index_add_(0, inverse_indices, outer_products_centered)

        ssd_multi = sum_of_outer_products_centered[multi_point_mask]
        counts_multi = counts[multi_point_mask]

        voxel_covariances[multi_point_mask] = ssd_multi / (counts_multi - 1).unsqueeze(-1).unsqueeze(-1)


    return voxel_means, voxel_covariances
