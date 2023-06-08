import numpy as np
import cv2
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

    def set_ngp_model(self, model):
        self.ngp_model = model

    def scale_pose(self, pose):
        scaled_pose = pose
        scaled_pose[:, 3] /= 2 * self.scale
        return scaled_pose

    def show_paras(self):
        print("dataset: ", self.dataset)
        print("Intrinsic: ", self.K)


