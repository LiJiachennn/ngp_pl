import numpy as np
import cv2
import torch

from tracker import Tracker
import tracker_utils

from models.rendering import render
from datasets.ray_utils import get_rays


class TrackerDirect(Tracker):
    def __init__(self, dataset):
        super().__init__(dataset)

    def esitmate_pose(self):
        # estimation pose under multi level
        for i in range(1):
            self.run_iteration(0)
        return

    def run_iteration(self, level):

        # get paras under current level
        img = self.imgPyramid[level]
        depth = self.depthPyramid[level]
        K = self.K[level]
        Kinv = np.linalg.inv(K)
        img_wh = self.img_wh[level]
        directions = self.directions[level]

        # cur pose
        pose_obj2cam = self.get_pose_obj2cam()
        pose_cam2obj = np.linalg.inv(pose_obj2cam)
        pose_cam2obj_scaled = self.scale_pose(pose_cam2obj)

        # use ngp model to render the image
        rays_o, rays_d = get_rays(directions, torch.from_numpy(pose_cam2obj_scaled).to(self.device))
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
        hJT = np.zeros((6, 1))
        H = np.zeros((6, 6))

        # prepare the gray img
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_render = cv2.cvtColor(rgb_render, cv2.COLOR_BGR2GRAY)

        # find the valid point
        valid_indices = np.where(opacity_render > 0.95)
        valid_indices = np.array(valid_indices).transpose()

        for i in range(0, valid_indices.shape[0], 1):
            # get the 2D point in the image
            uv = np.array([valid_indices[i][1], valid_indices[i][0]])
            if (uv[0]<1 or uv[0]>(img_wh[0]-2) or uv[1]<1 or uv[1]>(img_wh[1]-2)):
                continue

            # compute the depth
            Z = depth_render[uv[1]][uv[0]]

            # compute the 3D point
            XYZ = self.compute_3Dpoint(uv, Z, Kinv)

            # compute h for one point
            h = self.compute_color_diff(uv, gray_img, gray_render)
            # h may suffer from the border (eg. 128-0)
            # if abs(h) > 127:
            #     continue

            # compute J, JT and H for one point
            J_ = self.compute_Jacobian(K, uv, XYZ, gray_img)
            JT_ = J_.transpose()
            H_ = np.dot(JT_, J_)

            # sum
            hJT = hJT + h * JT_
            H = H + H_

        # compute delta xi
        # inverse or pseudo-inverse (inv or pinv)
        delta_xi = -np.dot(np.linalg.inv(H), hJT)
        delta_pose = self.exp(delta_xi).astype(np.float32)

        # update the pose
        self.set_pose_obj2cam(np.dot(delta_pose, pose_obj2cam))

        # check the results
        # cv2.imshow("img_level", img)
        # cv2.imshow("rgb_render", rgb_render)
        # depth_render_normalize = tracker_utils.depth2pseudo(depth_render, depth_min=0.4)
        # cv2.imshow("depth_render_normalize", depth_render_normalize)
        # opacity_render_normalize = (opacity_render*255).astype(np.uint8)
        # cv2.imshow("opacity_render", opacity_render_normalize)
        # cv2.waitKey(0)


    def compute_color_diff(self, uv, img, render):
        r = uv[1]
        c = uv[0]
        return float(img[r][c]) - float(render[r][c])

    def compute_Jacobian(self, K, uv, XYZ, gray_img):
        J = np.zeros((1, 6))

        de_dxy = self.de_dxy(gray_img, uv[0], uv[1])
        dxy_dXYZ = self.dxy_dXYZ(K, XYZ)
        dXYZ_dxi = self.dXYZ_dxi(XYZ)

        J = np.dot(np.dot(de_dxy, dxy_dXYZ), dXYZ_dxi)
        J = J.reshape(1, 6)
        return J