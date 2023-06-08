import numpy as np
import cv2
import torch

from tracker import Tracker
import tracker_utils

from models.rendering import render
from datasets.ray_utils import get_rays

from einops import rearrange


class TrackerDirect(Tracker):
    def __init__(self, dataset):
        super().__init__(dataset)

    def esitmation_pose(self):
        # estimation pose under multi level
        # for i in range(4):
        #     self.run_iteration(2)
        for i in range(2):
            self.run_iteration(1)
        for i in range(1):
            self.run_iteration(0)

    def run_iteration(self, level):

        # get paras under current level
        img = self.imgPyramid[level]
        K = self.K[level]
        img_wh = self.img_wh[level]
        directions = self.directions[level]

        # cur pose
        pose_obj2cam = self.get_pose_obj2cam()
        pose_cam2obj = np.linalg.inv(pose_obj2cam)
        pose_cam2obj_scaled = self.scale_pose(pose_cam2obj)

        # use ngp model to render the image
        rays_o, rays_d = get_rays(directions, torch.from_numpy(pose_cam2obj_scaled).cuda())
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

        # optimize the pose
        Jacobian = np.zeros((6, 1))
        Hession = np.zeros((6, 6))

        # find the valid point
        valid_indices = np.where(opacity_render > 0.95)
        valid_indices = np.array(valid_indices).transpose()

        for i in range(valid_indices.shape[0]):
            r = valid_indices[i][0]
            c = valid_indices[i][1]

            # compute Jacobian for one point
            h = self.compute_color_diff()
            J = self.compute_Jacobian()






        # check the results
        # cv2.imshow("img_level", img)
        # cv2.imshow("rgb_render", rgb_render)
        # depth_render_normalize = tracker_utils.depth2pseudo(depth_render, depth_min=0.4)
        # cv2.imshow("depth_render_normalize", depth_render_normalize)
        # opacity_render_normalize = (opacity_render*255).astype(np.uint8)
        # cv2.imshow("opacity_render", opacity_render_normalize)
        # cv2.waitKey(0)

        return

    def compute_color_diff(self):
        return

    def compute_Jacobian(self):
        return