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
        img_tensor = torch.from_numpy(img).cuda()
        depth_tensor = torch.from_numpy(np.float32(depth)).cuda()

        # cur pose
        pose_obj2cam = self.get_pose_obj2cam()
        pose_cam2obj = np.linalg.inv(pose_obj2cam)
        pose_cam2obj_scaled = self.scale_pose(pose_cam2obj)

        # to se3
        pose_cam2obj_se3 = self.ln(pose_cam2obj_scaled)
        pose_cam2obj_se3 = torch.from_numpy(np.float32(pose_cam2obj_se3)).cuda()
        pose_para_list_so3 = Variable(pose_cam2obj_se3[:3], requires_grad=True)
        pose_para_list_T = Variable(pose_cam2obj_se3[3:], requires_grad=True)

        # set optimizer
        optimizer_camera = torch.optim.Adam([{'params': pose_para_list_so3, 'lr': self.cam_lr * 0.2},
                                             {'params': pose_para_list_T, 'lr': self.cam_lr}])

        # set loss
        mse_loss = torch.nn.MSELoss()

        for i in range(self.steps):

            optimizer_camera.zero_grad()

            # convert cam paras of optimizer to RT of NGP
            pose_cur = self.get_pose_from_tensor(pose_para_list_so3, pose_para_list_T)

            # use ngp model to render the image
            rays_o, rays_d = get_rays(directions, torch.from_numpy(pose_cur).cuda())
            results_render = render(self.ngp_model, rays_o, rays_d,
                                    **{'test_time': True,
                                       'T_threshold': 1e-2,
                                       'exp_step_factor': 1 / 256})

            rgb_render = results_render['rgb'].reshape(img_wh[1], img_wh[0], 3).cpu().numpy()
            rgb_render = (rgb_render * 255).astype(np.uint8)
            rgb_render = cv2.cvtColor(rgb_render, cv2.COLOR_RGB2BGR)
            depth_render = results_render['depth'].reshape(img_wh[1], img_wh[0]).cpu().numpy()
            depth_render = self.scale_depth(depth_render)
            opacity_render = results_render['opacity'].reshape(img_wh[1], img_wh[0]).cpu().numpy()

            # to cuda
            img_render_tensor = torch.from_numpy(rgb_render).cuda()
            depth_render_tensor = torch.from_numpy(np.float32(depth_render)).cuda()

            # loss
            loss = mse_loss(torch.ones(5), torch.ones(5))

            # back prop
            loss.backward()
            optimizer_camera.step()
            optimizer_camera.zero_grad()

            print("step: ", i)
            print(pose_cam2obj_se3)
            print(cam_para_list_so3)
            print(cam_para_list_T)
            print(depth_tensor[290][330])

    def get_pose_from_tensor(self, so3_, T_):
        so3 = so3_.clone().detach().cpu().numpy()
        T = T_.clone().detach().cpu().numpy()

        xi = np.concatenate((so3, T))
        pose = self.exp(xi)
        return np.float32(pose)