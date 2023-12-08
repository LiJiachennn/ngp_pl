import numpy as np
import cv2

import torch
from torch.autograd import Variable

from tracker import Tracker
import tracker_utils

from models.rendering import render
from datasets.ray_utils import get_rays

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class TrackerMLP(Tracker):
    def __init__(self, dataset):
        super().__init__(dataset)

        # set some globle paras
        self.steps = 50
        self.sample_points = 1000
        self.cam_lr = 0.001

        self.seperate_LR = False

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

        # flatten
        img_tensor = img_tensor.view(-1, 3)
        depth_tensor = depth_tensor.view(-1)

        # cur pose
        pose_obj2cam = self.get_pose_obj2cam()
        pose_cam2obj = np.linalg.inv(pose_obj2cam)
        pose_cam2obj_scaled = self.down_scale_pose(pose_cam2obj)

        # to quaternion and T
        camera_tensor = self.get_tensor_from_camera(pose_cam2obj_scaled[:3, :])

        # set optimizer
        if self.seperate_LR:
            camera_tensor = camera_tensor.to(self.device).detach()
            quad = camera_tensor[:4]
            T = camera_tensor[-3:]
            quad = Variable(quad, requires_grad=True)
            T = Variable(T, requires_grad=True)
            cam_para_list_quad = [quad]
            cam_para_list_T = [T]
            optimizer_camera = torch.optim.Adam([{'params': cam_para_list_quad, 'lr': self.cam_lr * 5},
                                                 {'params': cam_para_list_T, 'lr': self.cam_lr}])
        else:
            camera_tensor = Variable(camera_tensor.to(self.device), requires_grad=True)
            cam_para_list = [camera_tensor]
            optimizer_camera = torch.optim.Adam(cam_para_list, lr=self.cam_lr)

        for i in range(self.steps):

            if self.seperate_LR:
                camera_tensor = torch.cat([quad, T], 0).to(self.device)

            optimizer_camera.zero_grad()

            # convert cam paras of optimizer to RT of NGP
            pose_cur = self.get_camera_from_tensor(camera_tensor).to(self.device)

            # use ngp model to render the image
            rays_o, rays_d = get_rays(directions, pose_cur)

            # select rays, need further accelerate
            random_indices = self.select_rays(pose_cur)

            # break
            if (random_indices.shape[0] < self.sample_points/5):
                continue

            # render rays
            results_render = render(self.ngp_model, rays_o[random_indices], rays_d[random_indices],
                                    **{'test_time': False})

            rgb_render_tensor = results_render['rgb']
            depth_render_tensor = results_render['depth']

            # conver depth render to real value
            depth_render_tensor = torch.mul(depth_render_tensor, 2 * self.scale * 1000)

            # loss
            # remove the invalid region
            # diff = abs(depth_tensor[random_indices] - depth_render_tensor)
            # depth_loss_mask = diff < 100

            loss = torch.mean(torch.abs(depth_tensor[random_indices] - depth_render_tensor))

            # back prop
            loss.backward()
            optimizer_camera.step()
            optimizer_camera.zero_grad()

            # print("step: ", i)
            # print(loss)
            # print(camera_tensor)

        # update the pose to tracker
        pose_opt = self.get_camera_from_tensor(camera_tensor).detach().cpu().numpy()
        pose_opt = self.up_scale_pose(pose_opt)
        pose_opt = np.vstack((pose_opt, np.array([0, 0, 0, 1])))
        pose_opt = np.linalg.inv(pose_opt)
        self.set_pose_obj2cam(np.float32(pose_opt))

        # check result
        # print(loss)


    def select_rays(self, pose_cur):
        # use ngp model to render the image, for select rays
        rays_o, rays_d = get_rays(self.directions[0], pose_cur)
        results_render = render(self.ngp_model, rays_o, rays_d,
                                **{'test_time': True,
                                   'T_threshold': 1e-2,
                                   'exp_step_factor': 1 / 256})

        # find the valid point
        opacity_render = results_render['opacity'].reshape(self.img_wh[0][1], self.img_wh[0][0]).cpu().numpy()
        valid_indices = np.where(opacity_render > 0.95)
        valid_indices = np.array(valid_indices).transpose()

        # flatten the indices
        flatten_indices = np.zeros(valid_indices.shape[0])
        flatten_indices = valid_indices[:, 0]*self.img_wh[0][0] + valid_indices[:, 1]

        # select part rays
        if (flatten_indices.shape[0] > self.sample_points):
            random_indices = np.array(np.random.choice(flatten_indices, size=self.sample_points, replace=False))
        else:
            random_indices = flatten_indices

        # visulize the result
        # tracking_result = self.imgPyramid[0].copy()
        # for i in range(random_indices.shape[0]):
        #     r = int(random_indices[i] / self.img_wh[0][0])
        #     c = int(random_indices[i] % self.img_wh[0][0])
        #     tracking_result[r][c] = 0.5 * np.array([255, 0, 255]) + 0.5 * self.imgPyramid[0][r][c]
        # cv2.imshow("tracking_result", tracking_result)
        # cv2.waitKey(0)

        return random_indices

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