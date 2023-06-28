import numpy as np
import cv2

import torch
from torch.autograd import Variable

from tracker import Tracker
import tracker_utils

from models.rendering import render
from datasets.ray_utils import get_rays

class TrackerMLP(Tracker):
    def __init__(self, dataset):
        super().__init__(dataset)

        # set some globle paras
        self.steps = 1
        self.sample_points = 1000
        self.cam_lr = 0.001

    def esitmate_pose(self):
        self.run_level(0)

    def run_level(self, level):

        # get paras under current level
        img = self.imgPyramid[level]
        depth = self.depthPyramid[level]
        K = self.K[level]
        Kinv = np.linalg.inv(K)
        img_wh = self.img_wh[level]
        directions = self.directions[level]

        # to cuda
        img_tensor = torch.from_numpy(img).to(self.device)
        depth_tensor = torch.from_numpy(np.float32(depth)).to(self.device)

        # cur pose
        pose_obj2cam = self.get_pose_obj2cam()
        pose_cam2obj = np.linalg.inv(pose_obj2cam)
        pose_cam2obj_scaled = self.scale_pose(pose_cam2obj)

        # to quaternion and T
        camera_tensor = self.get_tensor_from_camera(pose_cam2obj_scaled)
        camera_tensor = Variable(camera_tensor.to(self.device), requires_grad=True)
        cam_para_list = [camera_tensor]

        # set optimizer
        optimizer_camera = torch.optim.Adam(cam_para_list, lr=self.cam_lr)

        # set loss
        mse_loss = torch.nn.MSELoss()

        for i in range(self.steps):

            optimizer_camera.zero_grad()

            # convert cam paras of optimizer to RT of NGP
            pose_cur = self.get_camera_from_tensor(camera_tensor).to(self.device)

            # use ngp model to render the image
            rays_o, rays_d = get_rays(directions, pose_cur)
            results_render = render(self.ngp_model, rays_o, rays_d,
                                    **{'test_time': True,
                                       'T_threshold': 1e-2,
                                       'exp_step_factor': 1 / 256})

            rgb_render_tensor = results_render['rgb'].reshape(img_wh[1], img_wh[0], 3)
            depth_render_tensor = results_render['depth'].reshape(img_wh[1], img_wh[0])
            # rgb_render = results_render['rgb'].reshape(img_wh[1], img_wh[0], 3).cpu().numpy()
            # rgb_render = (rgb_render * 255).astype(np.uint8)
            # rgb_render = cv2.cvtColor(rgb_render, cv2.COLOR_RGB2BGR)
            # depth_render = results_render['depth'].reshape(img_wh[1], img_wh[0]).cpu().numpy()
            # depth_render = self.scale_depth(depth_render)
            # opacity_render = results_render['opacity'].reshape(img_wh[1], img_wh[0]).cpu().numpy()

            # loss
            loss = mse_loss(depth_tensor, depth_render_tensor)

            # back prop
            loss.backward()
            optimizer_camera.step()
            optimizer_camera.zero_grad()

            print("step: ", i)
            print(pose_cam2obj_se3)
            print(cam_para_list_so3)
            print(cam_para_list_T)
            print(depth_tensor[290][330])

    def quad2rotation(self, quad):
        """
        Convert quaternion to rotation. Since all operation in pytorch, support gradient passing.

        Args:
            quad (tensor, batch_size*4): quaternion.

        Returns:
            rot_mat (tensor, batch_size*3*3): rotation.
        """
        bs = quad.shape[0]
        qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
        two_s = 2.0 / (quad * quad).sum(-1)
        rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
        rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
        rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
        rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
        rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
        rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
        rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
        rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
        rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
        rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
        return rot_mat

    def get_camera_from_tensor(self, inputs):
        """
        Convert quaternion and translation to transformation matrix.
        """
        N = len(inputs.shape)
        if N == 1:
            inputs = inputs.unsqueeze(0)
        quad, T = inputs[:, :4], inputs[:, 4:]
        R = self.quad2rotation(quad)
        RT = torch.cat([R, T[:, :, None]], 2)
        if N == 1:
            RT = RT[0]
        return RT

    def get_tensor_from_camera(self, RT, Tquad=False):
        """
        Convert transformation matrix to quaternion and translation.
        """
        gpu_id = -1
        if type(RT) == torch.Tensor:
            if RT.get_device() != -1:
                RT = RT.detach().cpu()
                gpu_id = RT.get_device()
            RT = RT.numpy()
        from mathutils import Matrix
        R, T = RT[:3, :3], RT[:3, 3]
        rot = Matrix(R)
        quad = rot.to_quaternion()
        if Tquad:
            tensor = np.concatenate([T, quad], 0)
        else:
            tensor = np.concatenate([quad, T], 0)
        tensor = torch.from_numpy(tensor).float()
        if gpu_id != -1:
            tensor = tensor.to(gpu_id)
        return tensor