import numpy as np
import cv2
import torch

from tracker import Tracker
import tracker_utils

from models.rendering import render
from datasets.ray_utils import get_rays

class TrackerMLP(Tracker):
    def __init__(self, dataset):
        super().__init__(dataset)


    def esitmation_pose(self):



        for i in range(1):
            self.run_iteration(0)
        return