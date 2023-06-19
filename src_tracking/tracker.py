import numpy as np
import cv2
import torch
import scipy

from models.rendering import render
from datasets.ray_utils import get_rays
from datasets.ray_utils import get_ray_directions


class Tracker():
    def __init__(self, dataset):
        self.dataset = dataset
        self.set_intrinsics()
        # self.show_paras()

    def set_intrinsics(self):
        self.K = []
        self.img_wh = []
        self.directions = []

        if self.dataset == 'rbot':
            fx = 650.048
            fy = 647.183
            cx = 324.328
            cy = 257.323
            w = 640
            h = 512
            self.K.append(np.float32([[fx, 0, cx],
                                      [0, fy, cy],
                                      [0,  0,  1]]))
            self.img_wh.append((w, h))
        else:
            raise ValueError(f'{dataset} is not support now!')

        # compute rays
        self.scale = 1.05  # consistent with para when training
        self.directions.append(get_ray_directions(self.img_wh[0][1], self.img_wh[0][0], self.K[0]).cuda())

        # multi level, append level 1 and level 2
        for i in range(1, 3):
            self.K.append(self.K[0] / pow(2, i))
            w_ = int(self.img_wh[0][0] / pow(2, i))
            h_ = int(self.img_wh[0][1] / pow(2, i))
            self.img_wh.append((w_, h_))
            self.directions.append(get_ray_directions(self.img_wh[i][1], self.img_wh[i][0], self.K[i]).cuda())

    def set_pose_obj2cam(self, pose):
        self.pose_obj2cam = pose

    def get_pose_obj2cam(self):
        return self.pose_obj2cam

    def set_image(self, img):
        self.imgPyramid = []
        self.imgPyramid.append(img)
        for i in range(1, 3):
            img_ = cv2.resize(img, self.img_wh[i], cv2.INTER_LINEAR)
            self.imgPyramid.append(img_)

    def set_depth(self, depth):
        self.depthPyramid = []
        self.depthPyramid.append(depth)
        for i in range(1, 3):
            depth_ = cv2.resize(depth, self.img_wh[i], cv2.INTER_NEAREST)
            self.depthPyramid.append(depth_)

    def set_ngp_model(self, model):
        self.ngp_model = model

    def scale_pose(self, pose):
        scaled_pose = pose.copy()
        scaled_pose[:, 3] /= 2 * self.scale
        return scaled_pose

    def scale_depth(self, depth):
        depth *= 2 * self.scale
        return depth

    def visulize_tracking_result(self):
        tracking_result = self.imgPyramid[0].copy()

        # cur pose
        pose_obj2cam = self.get_pose_obj2cam()
        pose_cam2obj = np.linalg.inv(pose_obj2cam)
        pose_cam2obj_scaled = self.scale_pose(pose_cam2obj)

        # use ngp model to render the image
        rays_o, rays_d = get_rays(self.directions[0], torch.from_numpy(pose_cam2obj_scaled).cuda())
        results_render = render(self.ngp_model, rays_o, rays_d,
                                **{'test_time': True,
                                   'T_threshold': 1e-2,
                                   'exp_step_factor': 1 / 256})

        rgb_render = results_render['rgb'].reshape(self.img_wh[0][1], self.img_wh[0][0], 3).cpu().numpy()
        rgb_render = (rgb_render * 255).astype(np.uint8)
        rgb_render = cv2.cvtColor(rgb_render, cv2.COLOR_RGB2BGR)
        opacity_render = results_render['opacity'].reshape(self.img_wh[0][1], self.img_wh[0][0]).cpu().numpy()

        # find the valid point
        valid_indices = np.where(opacity_render > 0.95)
        valid_indices = np.array(valid_indices).transpose()

        # visulize the result
        for i in range(valid_indices.shape[0]):
            r = valid_indices[i][0]
            c = valid_indices[i][1]
            # render texture
            # tracking_result[r][c] = 0.5 * rgb_render[r][c] + 0.5 * self.imgPyramid[0][r][c]
            # render single color
            tracking_result[r][c] = 0.5 * np.array([255, 0, 255]) + 0.5 * self.imgPyramid[0][r][c]

        return tracking_result

    def compute_3Dpoint(self, uv, depth, Kinv):
        X = depth * (Kinv[0][0] * uv[0] + Kinv[0][2])
        Y = depth * (Kinv[1][1] * uv[1] + Kinv[1][2])
        return np.array([X, Y, depth])

    def de_dxy(self, img, u_, v_):
        l = u_ - 1
        r = u_ + 1
        u = v_ - 1
        d = v_ + 1
        dx = (img[v_][r] - img[v_][l]) / 2.0
        dy = (img[d][u_] - img[u][u_]) / 2.0
        return np.array([dx, dy])

    def dxy_dXYZ(self, K, XYZ):
        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]
        X, Y, Z = XYZ[0], XYZ[1], XYZ[2]
        dxy_dXYZ = np.array([[fx / Z, 0,      -(fx * X) / (Z * Z)],
                             [0,      fy / Z, -(fy * Y) / (Z * Z)]])
        return dxy_dXYZ

    def dXYZ_dxi(self, XYZ):
        X, Y, Z = XYZ[0], XYZ[1], XYZ[2]
        dXYZ_dxi = np.array([[0, Z, -Y, 1, 0, 0],
                             [-Z, 0, X, 0, 1, 0],
                             [Y, -X, 0, 0, 0, 1]])
        return dXYZ_dxi

    def skew_symmetric(self, vector):
        matrix = np.zeros((3, 3))
        matrix[0, 1] = -vector[2]
        matrix[0, 2] = vector[1]
        matrix[1, 0] = vector[2]
        matrix[1, 2] = -vector[0]
        matrix[2, 0] = -vector[1]
        matrix[2, 1] = vector[0]
        return matrix

    def exp(self, xi):
        se3 = np.zeros((4, 4))
        se3[:3, :3] = self.skew_symmetric(xi[:3])
        se3[:3, 3] = xi[3:].reshape(3)
        pose = scipy.linalg.expm(se3)
        return pose

    def show_paras(self):
        print("dataset: ", self.dataset)
        print("Intrinsic: ", self.K)


