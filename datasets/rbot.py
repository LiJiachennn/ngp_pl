import torch
import glob
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image, read_image_with_mask

from .base import BaseDataset


class RBOTDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.shift = 0
            self.scale = 1.05 # enlarge a little
            self.read_meta(split)

    def read_intrinsics(self):
        fx = 650.048 * self.downsample
        fy = 647.183 * self.downsample
        cx = 324.328 * self.downsample
        cy = 257.323 * self.downsample
        w = int(640 * self.downsample)
        h = int(512 * self.downsample)

        K = np.float32([[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]])

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        # test on rbot, set variant and obj
        variant = 'a_regular'
        obj = 'cat'

        # split data, index: [begin, end)
        data_split = [200, 900, 1000]
        if split == 'train':
            begin, end = 0, data_split[0]+1
        elif split == 'trainval':
            begin, end = 0, data_split[1]+1
        elif split == 'trainvaltest':
            begin, end = 0, data_split[2]+1
        elif split == 'val':
            begin, end = data_split[0], data_split[1]+1
        elif split == 'test':
            begin, end = data_split[1], data_split[2]+1
        else: raise ValueError(f'{split} split not recognized!')

        # set poses and imgs path
        poses_path = os.path.join(self.root_dir, 'poses_first_mat.txt')
        img_paths = sorted(glob.glob(os.path.join(self.root_dir, obj, 'frames', variant + '*.png')))
        mask_paths = sorted(glob.glob(os.path.join(self.root_dir, obj, 'mask', '*.png')))
        # load poses
        poses = np.array(self.load_pose_rbot(poses_path))

        # get the splited data (poses and img)
        poses = poses[begin:end, :, :]
        img_paths = img_paths[begin:end]
        mask_paths = mask_paths[begin:end]

        if len(img_paths) != poses.shape[0]:
            raise ValueError(f'len(img_paths) != poses.shape[0], pleas check.')

        print(f'Loading {len(img_paths)} {split} images ...')
        for i in tqdm(range(poses.shape[0])):
            w2c = poses[i]
            c2w = self.pose_inverse(w2c)
            c2w[:, 3] -= self.shift
            c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
            self.poses += [c2w]

            img = read_image_with_mask(img_paths[i], mask_paths[i], self.img_wh)
            self.rays += [img]

        self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)

    def load_pose_rbot(self, path):
        poses = np.loadtxt(path)
        poses_ = []
        for i in range(poses.shape[0]):
            pose_ = np.zeros((3, 4))
            pose_[:, 0:3] = poses[i][0:9].reshape(3, 3)
            pose_[:, 3] = poses[i][9:12] * 0.001
            poses_.append(pose_)
        return poses_


    def pose_inverse(self, pose):
        padded_pose = np.pad(pose, ((0,1), (0,0)), mode='constant')
        padded_pose[3] = [0, 0, 0, 1]
        pose_inv = np.linalg.inv(padded_pose)
        return pose_inv[:3]