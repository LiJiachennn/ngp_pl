import numpy as np
import cv2

import glob
import os

from tqdm import tqdm
from tracker_direct import TrackerDirect
import tracker_utils

from models.networks import NGP
from utils import load_ckpt

import warnings; warnings.filterwarnings("ignore")


def track_one_video(root_path, NGP_model_path, variant, object):

    # load the pretrained NGP model
    ngp_model = NGP(scale=0.5).cuda()
    load_ckpt(ngp_model, NGP_model_path + object + '_trainvaltest_50_slim.ckpt')

    # load imgs
    img_paths = sorted(glob.glob(os.path.join(root_path, object, 'frames', variant + '*.png')))

    # load depth imgs
    depth_paths = sorted(glob.glob(os.path.join(root_path, object, 'depth', '*.png')))

    # load poses
    poses_path = root_path + "poses_first_mat.txt"
    poses_gt = tracker_utils.load_pose_rbot(poses_path)

    # set tracker
    tracker = TrackerDirect('rbot')
    tracker.set_ngp_model(ngp_model)
    tracker.set_pose_obj2cam(poses_gt[0])

    # compute result
    correct_num = 0

    for i in tqdm(range(len(img_paths))):
        # load img
        img = cv2.imread(img_paths[i], cv2.IMREAD_UNCHANGED)
        tracker.set_image(img)

        # load depth
        depth = cv2.imread(depth_paths[i], cv2.IMREAD_UNCHANGED)
        tracker.set_depth(depth)

        # set gt pose, for testing
        # tracker.set_pose_obj2cam(poses_gt[i])

        # estimate the pose
        tracker.esitmation_pose()
        pose_pred = tracker.get_pose_obj2cam()

        # visulize the tracking result
        result_tracking = tracker.visulize_tracking_result()

        # compute error
        error_R, error_t = tracker_utils.compute_error_pose(pose_pred, poses_gt[i])
        success = tracker_utils.compute_success_ncm_ndegree(error_R, error_t, 5.0, 0.05)
        if success:
            correct_num += 1
        else:
            tracker.set_pose_obj2cam(poses_gt[i])

        cv2.imshow("result", result_tracking)
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == 27:
            break

    accuracy = (correct_num - 1) / (len(img_paths) - 1)
    return accuracy


def evaluate_rbot():
    print("evaluate rbot dataset.")

    root_path = "/data/DATASETS/RBOT_dataset_2/"
    NGP_model_path = "/data/codes/ngp_pl/pretrained_models/rbot/"

    variants = ['a_regular', 'b_dynamiclight', 'c_noisy', 'd_occlusion']
    objects = ['ape', 'bakingsoda', 'benchviseblue', 'broccolisoup', 'cam',
               'can', 'cat', 'clown', 'cube', 'driller',
               'duck', 'eggbox', 'glue', 'iron', 'koalacandy', 'lamp',
               'phone', 'squirrel']

    results = []
    for v in range(0, 1):
        for o in range(6, 7):
            result = track_one_video(root_path, NGP_model_path, variants[v], objects[o])
            results.append(result)

    # show results
    for r in range(len(results)):
        print(results[r] * 100)



if __name__ == '__main__':
    evaluate_rbot()


