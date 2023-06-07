import numpy as np
import cv2
import torch

from tracker import Tracker

from models.rendering import render
from datasets.ray_utils import get_rays

from einops import rearrange


class TrackerDirect(Tracker):
    def __init__(self, dataset):
        super().__init__(dataset)

    def run_iteration(self, level):

        # cur pose
        pose_obj2cam = self.get_pose_obj2cam().astype(np.float32)
        pose_cam2obj = np.linalg.inv(pose_obj2cam)
        pose_cam2obj_scaled = self.scale_pose(pose_cam2obj)

        # use ngp model to render the image
        rays_o, rays_d = get_rays(self.directions, torch.from_numpy(pose_cam2obj_scaled).cuda())
        results_render = render(self.ngp_model, rays_o, rays_d,
                                **{'test_time': True,
                                   'T_threshold': 1e-2,
                                   'exp_step_factor': 1 / 256})

        img_render = results_render['rgb'].reshape(self.img_wh[1], self.img_wh[0], 3).cpu().numpy()
        img_render = (img_render * 255).astype(np.uint8)
        img_render = cv2.cvtColor(img_render, cv2.COLOR_RGB2BGR)
        cv2.imshow("img_render", img_render)

        return

    def esitmation_pose(self):
        for i in range(1):
            self.run_iteration(0)