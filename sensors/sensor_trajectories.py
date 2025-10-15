import torch
import numpy as np
import torch.nn as nn
import pytorch3d.transforms

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation


class LidarPose:
    def __init__(self, lidar_rotation, lidar_translation, timestamp):
        rotation_quat = pytorch3d.transforms.matrix_to_quaternion(torch.tensor(lidar_rotation)).float().to('cuda')
        self.lidar_rotation = torch.nn.Parameter(rotation_quat, requires_grad=False)
        self.lidar_translation = torch.nn.Parameter(torch.Tensor(lidar_translation).float().to('cuda'), requires_grad=False)
        self.timestamp = timestamp

class SensorTrajectories:
    def __init__(self, lidar_rotations, lidar_translations, lidar_timestamps, pose_cl, initialized_pose_cl = None):
        assert len(lidar_rotations) == len(lidar_translations)
        assert len(lidar_rotations) == len(lidar_timestamps)


        self.lidar_poses = []
        for i in range(len(lidar_rotations)):
            self.lidar_poses.append(LidarPose(lidar_rotations[i], lidar_translations[i], lidar_timestamps[i]))

        rotation_cl = pose_cl[:3,:3]
        translation_cl = pose_cl[:3,3]
        rotation_cl_quat = pytorch3d.transforms.matrix_to_quaternion(torch.tensor(rotation_cl))
        # rotation_disturbance = pytorch3d.transforms.axis_angle_to_quaternion(torch.rand(3) * 0.0)
        # translation_disturbance = np.random.rand(3) * 0.5

        self.gt_translation_cl = torch.Tensor(translation_cl)
        self.gt_rotation_cl = rotation_cl_quat


        self.gt_rotation_cl = rotation_cl_quat.float().to('cuda')
        self.gt_translation_cl = torch.Tensor(translation_cl).float().to('cuda')
        #TODO: Wrong in compute pose error
        # initial_rotation = torch.Tensor([[0., -1., 0.], [0., 0., -1.], [1., 0., 0,]]).float().to('cuda')
        # initial_translation = torch.Tensor([0., 0., 0.]).float().to('cuda')

        # initial_rotation = torch.Tensor([[4.27303574e-04, -9.99121606e-01,  4.19031493e-02], [-7.18921563e-03, -4.19051386e-02, -9.99095678e-01], [9.99974072e-01,  1.25666411e-04, -7.20080640e-03 ]]).float().to('cuda')


        rotation_disturbance = pytorch3d.transforms.matrix_to_quaternion(pytorch3d.transforms.euler_angles_to_matrix(torch.ones(3).float().to('cuda') * 3 / 180 * 1.57, convention='XYZ'))
        # initial_rotation_quat = pytorch3d.transforms.matrix_to_quaternion(initial_rotation)

        if initialized_pose_cl is None:
            initial_rotation_mat = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
            initial_rotation_mat = torch.Tensor(initial_rotation_mat).float().to('cuda')
            initial_translation = self.gt_translation_cl * 0.
        else:
            initial_rotation_mat = torch.Tensor(initialized_pose_cl[:3,:3]).float().to('cuda')
            initial_translation = torch.Tensor(initialized_pose_cl[:3,3]).float().to('cuda')
        # initial_rotation_quat = self.gt_rotation_cl
        # initial_translation = self.gt_translation_cl
        # initial_rotation_quat = pytorch3d.transforms.quaternion_multiply(initial_rotation_quat, rotation_disturbance)

        initial_rotation_quat = pytorch3d.transforms.matrix_to_quaternion(initial_rotation_mat)
        # initial_translation = self.gt_translation_cl * 0.

        self.rotation_cl = initial_rotation_quat
        self.translation_cl = initial_translation

        rotation_cl_delta = pytorch3d.transforms.matrix_to_quaternion(torch.eye(3)).float().to('cuda')
        translation_cl_delta = torch.zeros(3).float().to('cuda')
        self.rotation_cl_delta = torch.nn.Parameter(rotation_cl_delta, requires_grad=False)
        self.translation_cl_delta = torch.nn.Parameter(translation_cl_delta, requires_grad=False)



    def get_extrinsics(self):
        rotation_cl = pytorch3d.transforms.quaternion_multiply(self.rotation_cl_delta, self.rotation_cl)
        translation_cl = pytorch3d.transforms.quaternion_apply(self.rotation_cl_delta, self.translation_cl) + self.translation_cl_delta
        return rotation_cl, translation_cl

    def update_extrinsics(self):
        with torch.no_grad():
            self.rotation_cl = pytorch3d.transforms.quaternion_multiply(self.rotation_cl_delta, self.rotation_cl)
            self.translation_cl = pytorch3d.transforms.quaternion_apply(self.rotation_cl_delta, self.translation_cl) + self.translation_cl_delta


            self.rotation_cl_delta.data = pytorch3d.transforms.matrix_to_quaternion(torch.eye(3)).float().to('cuda')
            self.translation_cl_delta.data = torch.zeros(3).float().to('cuda')
        return True

    def get_length(self):
        return len(self.lidar_poses)

    def start_optimize_cl(self):
        if not self.rotation_cl_delta.requires_grad:
            self.rotation_cl_delta.requires_grad_(True)
        if not self.translation_cl_delta.requires_grad:
            self.translation_cl_delta.requires_grad_(True)


    def start_optimize_rotation_cl(self):
        if not self.rotation_cl_delta.requires_grad:
            self.rotation_cl_delta.requires_grad_(True)
        return

    def stop_optimize_rotation_cl(self):
        self.rotation_cl_delta.requires_grad_(False)
        self.rotation_cl_delta.grad = None
        return

    def stop_optimize_translation_cl(self):
        self.translation_cl_delta.requires_grad_(False)
        self.translation_cl_delta.grad = None
        return

    def start_optimize_translation_cl(self):
        if not self.translation_cl_delta.requires_grad:
            self.translation_cl_delta.requires_grad_(True)

    def quaternion_euler_error(
            self,
            pred_q: torch.Tensor,
            gt_q: torch.Tensor,
            rotation_order: str = 'XYZ',
            eps: float = 1e-7
    ) -> torch.Tensor:
        """
        计算四元数之间的欧拉角三轴误差

        参数:
            pred_q (Tensor): 预测四元数 [..., 4] (w, x, y, z)
            gt_q (Tensor): 真值四元数 [..., 4]
            rotation_order (str): 欧拉角分解顺序，默认为'XYZ'
            eps (float): 数值稳定系数

        返回:
            error (Tensor): 欧拉角三轴绝对误差 [..., 3] (单位:弧度)
        """
        # 归一化四元数 (处理非单位四元数)
        pred_q = pred_q / (torch.norm(pred_q, dim=-1, keepdim=True) + eps)
        gt_q = gt_q / (torch.norm(gt_q, dim=-1, keepdim=True) + eps)

        # 转换为旋转矩阵
        pred_rot = pytorch3d.transforms.quaternion_to_matrix(pred_q)  # [..., 3, 3]
        gt_rot = pytorch3d.transforms.quaternion_to_matrix(gt_q)  # [..., 3, 3]

        # 转换为欧拉角
        pred_euler = pytorch3d.transforms.matrix_to_euler_angles(pred_rot, rotation_order)  # [..., 3]
        gt_euler = pytorch3d.transforms.matrix_to_euler_angles(gt_rot, rotation_order)  # [..., 3]

        # 计算周期性角度误差
        diff = pred_euler - gt_euler
        diff_rad = torch.remainder(diff + torch.pi, 2 * torch.pi) - torch.pi  # [-π, π]
        # 转换为角度制并取绝对值
        diff_deg = torch.abs(diff_rad) * (180.0 / torch.pi)  # [..., 3] (degree)

        return diff_deg

    def get_relative_pose_error(self):
        frame_num = len(self.lidar_poses)

        i = 5
        rotation_wc, translation_wc = self.get_camera_pose(i)
        next_rotation_wc, next_translation_wc = self.get_camera_pose(i+1)
        relative_rotation = (rotation_wc).T @ (next_rotation_wc)

        gt_rotation_wc, gt_translation_wc = self.get_camera_pose_gt(i)
        next_gt_rotation_wc, next_gt_translation_wc = self.get_camera_pose_gt(i+1)
        relative_rotation_gt = (gt_rotation_wc).T @ (next_gt_rotation_wc)
        rotation_err = relative_rotation.T @ relative_rotation_gt
        # relative_rotation_euler = pytorch3d.transforms.matrix_to_euler_angles(relative_rotation, convention='XYZ')
        # relative_rotation_euler_gt = pytorch3d.transforms.matrix_to_euler_angles(relative_rotation_gt, convention='XYZ')
        return pytorch3d.transforms.matrix_to_axis_angle(rotation_err)



    def compute_extrinsic_error(self, rotation_cl, translation_cl):
        # gt_rotation_matrix_cl = pytorch3d.transforms.quaternion_to_matrix(self.gt_rotation_cl.cpu()).numpy()
        # curr_rotation_matrix_cl = pytorch3d.transforms.quaternion_to_matrix(self.rotation_cl.detach().cpu()).numpy()
        euler_error = self.quaternion_euler_error(rotation_cl.detach().cpu(), self.gt_rotation_cl.detach().cpu())
        gt_translation_cl = self.gt_translation_cl.cpu().numpy()
        curr_translation_cl = translation_cl.detach().cpu().numpy()
        # relative_rotation_error = gt_rotation_matrix_cl @ curr_rotation_matrix_cl.T
        relative_translation_error = gt_translation_cl - curr_translation_cl
        return euler_error, torch.Tensor(relative_translation_error)

    def get_extrinsic_error(self):
        # gt_rotation_matrix_cl = pytorch3d.transforms.quaternion_to_matrix(self.gt_rotation_cl.cpu()).numpy()
        # curr_rotation_matrix_cl = pytorch3d.transforms.quaternion_to_matrix(self.rotation_cl.detach().cpu()).numpy()
        euler_error = self.quaternion_euler_error(self.rotation_cl.detach().cpu(), self.gt_rotation_cl.detach().cpu())
        gt_translation_cl = self.gt_translation_cl.cpu().numpy()
        curr_translation_cl = self.translation_cl.detach().cpu().numpy()
        # relative_rotation_error = gt_rotation_matrix_cl @ curr_rotation_matrix_cl.T
        relative_translation_error = gt_translation_cl - curr_translation_cl
        return euler_error, torch.Tensor(relative_translation_error)

    def training_setup(self, training_args):
        l = [
            {'params': [lidar_pose.lidar_rotation for lidar_pose in self.lidar_poses], 'lr': training_args.pose_rotation_lr,
             "name": "rotation_lidar"},
            {'params': [lidar_pose.lidar_translation for lidar_pose in self.lidar_poses], 'lr': training_args.pose_translation_lr,
             "name": "translation_lidar"},
            {'params': [self.rotation_cl_delta], 'lr': training_args.pose_rotation_lr,
             "name": "rotation_cl"},
            {'params': [self.translation_cl_delta], 'lr': training_args.pose_translation_lr,
             "name": "translation_cl"},
        ]
        self.l = l
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)



    def update_extrinsic_in_search(self, rotation_cl_quat, translation_cl):
        self.rotation_cl.data.copy_(rotation_cl_quat)
        self.translation_cl.data.copy_(translation_cl)
        self.rotation_cl_delta.data = pytorch3d.transforms.matrix_to_quaternion(torch.eye(3)).float().to('cuda')
        self.translation_cl_delta.data = torch.zeros(3).float().to('cuda')


    def update_learning_rate(self, ratio):
        initial_rotation_lr = self.l[2]['lr']
        initial_translation_lr = self.l[3]['lr']
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "rotation_cl":
                param_group['lr'] = initial_rotation_lr * ratio
            if param_group["name"] == "translation_cl":
                param_group['lr'] = initial_rotation_lr * ratio

    def get_all_lidar_poses(self):
        rotation_param_list = []
        translation_param_list = []
        for lidar_pose in self.lidar_poses:
            rotation_param_list.append(lidar_pose.lidar_rotation)
            translation_param_list.append(lidar_pose.lidar_translation)
        return rotation_param_list, translation_param_list

    def get_lidar_pose(self, frame_num):
        lidar_pose = self.lidar_poses[frame_num]
        rotation = lidar_pose.lidar_rotation
        translation = lidar_pose.lidar_translation
        return rotation, translation

    def get_camera_pose(self, frame_num):
        rotation_cl, translation_cl = self.get_extrinsics()
        rotation_wl, translation_wl = self.get_lidar_pose(frame_num)
        rotation_wl = pytorch3d.transforms.quaternion_to_matrix(rotation_wl)
        rotation_wc = rotation_wl @ pytorch3d.transforms.quaternion_to_matrix(rotation_cl).T
        #twc = Rwl*tlc+twl = Rwl*(-Rcl.T * tcl) + twl
        translation_wc = -(rotation_wl @ pytorch3d.transforms.quaternion_to_matrix(rotation_cl).T) @ translation_cl + translation_wl
        return rotation_wc, translation_wc

    def get_camera_pose_gt(self, frame_num):
        rotation_wl, translation_wl = self.get_lidar_pose(frame_num)
        rotation_wl = pytorch3d.transforms.quaternion_to_matrix(rotation_wl)
        rotation_wc = rotation_wl @ pytorch3d.transforms.quaternion_to_matrix(self.gt_rotation_cl).T
        #twc = Rwl*tlc+twl = Rwl*(-Rcl.T * tcl) + twl
        translation_wc = -(rotation_wl @ pytorch3d.transforms.quaternion_to_matrix(self.gt_rotation_cl).T) @ self.gt_translation_cl + translation_wl
        return rotation_wc, translation_wc

    # def get_gt_camera_pose(self, frame_num):
    #     rotation_wl, translation_wl = self.get_lidar_pose(frame_num)
    #     rotation_wl = pytorch3d.transforms.quaternion_to_matrix(rotation_wl).to(self.gt_rotation_cl.device)
    #     translation_wl = translation_wl.to(self.gt_translation_cl.device)
    #
    #     rotation_wc = rotation_wl @ pytorch3d.transforms.quaternion_to_matrix(self.gt_rotation_cl).T
    #     # twc = Rwl*tlc+twl = Rwl*(-Rcl.T * tcl) + twl
    #     translation_wc = -(rotation_wl @ pytorch3d.transforms.quaternion_to_matrix(
    #         self.gt_rotation_cl).T) @ self.gt_translation_cl + translation_wl
    #     return rotation_wc, translation_wc

if __name__ == '__main__':
    lidar_rotations = [li.SO3.exp(torch.Tensor(np.random.rand(3))).matrix()[:3,:3].numpy()]
    lidar_translations = [torch.Tensor(np.random.rand(3)).numpy()]

    pose_cl = li.SE3.exp(torch.Tensor(np.random.rand(6))).matrix().numpy()

    traj = SensorTrajectories(lidar_rotations, lidar_translations, [0.], pose_cl)
    rotation_wc, translation_wc = traj.get_camera_pose(0)
    pred_pose_wc = torch.eye(4)
    pred_pose_wc[:3,:3] = rotation_wc
    pred_pose_wc[:3,3] = translation_wc

    pose_wl = np.eye(4)
    pose_wl[:3,:3] = lidar_rotations[0]
    pose_wl[:3,3] = lidar_translations[0]


    pose_lc = np.linalg.inv(pose_cl)
    pose_wc = pose_wl @ pose_lc

    print(pose_wc)
    print(pred_pose_wc)

