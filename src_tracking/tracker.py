import numpy as np
import cv2


class Tracker():
    def __init__(self, dataset):
        self.dataset = dataset
        self.set_intrinsics()
        # self.show_paras()

    def set_intrinsics(self):
        if self.dataset == 'rbot':
            fx = 650.048
            fy = 647.183
            cx = 324.328
            cy = 257.323
            w = 640
            h = 512
            self.K = np.float32([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0,  0,  1]])
            self.img_wh = (w, h)
        else:
            raise ValueError(f'{dataset} is not support now!')

    def set_pose_obj2cam(self, pose):
        self.pose_obj2cam = pose

    def get_pose_obj2cam(self):
        return self.pose_obj2cam

    def show_paras(self):
        print("dataset: ", self.dataset)
        print("Intrinsic: ", self.K)


