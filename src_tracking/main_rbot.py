import numpy as np
import cv2

import glob
import os

from tqdm import tqdm
from tracker import Tracker
import tracker_utils

def track_one_video(root_path, variant, object):

    # load imgs
    img_paths = sorted(glob.glob(os.path.join(root_path, object, 'frames', variant + '*.png')))

    # load poses
    poses_path = root_path + "poses_first_mat.txt"
    poses_gt = np.array(tracker_utils.load_pose_rbot(poses_path))

    # set tracker
    tracker = Tracker('rbot')
    tracker.set_pose_obj2cam(poses_gt[0])

    # compute result
    correct_num = 0

    for i in tqdm(range(len(img_paths))):
        # load img
        img = cv2.imread(img_paths[i], 1)

        # set gt pose, for testing
        tracker.set_pose_obj2cam(poses_gt[i])

        # estimate the pose
        pose_pred = tracker.get_pose_obj2cam()

        # compute error
        error_R, error_t = tracker_utils.compute_error_pose(pose_pred, poses_gt[i])
        success = tracker_utils.compute_success_ncm_ndegree(error_R, error_t, 5.0, 50.0)
        correct_num += success

        cv2.imshow("img", img)
        key = cv2.waitKey(1)
        if key == 27:
            break

    accuracy = (correct_num - 1) / (len(img_paths) - 1)
    return accuracy


def evaluate_rbot():
    print("evaluate rbot dataset.")

    root_path = "/data/DATASETS/RBOT_dataset_2/"

    variants = ['a_regular', 'b_dynamiclight', 'c_noisy', 'd_occlusion']
    objects = ['ape', 'bakingsoda', 'benchviseblue', 'broccolisoup', 'cam',
               'can', 'cat', 'clown', 'cube', 'driller',
               'duck', 'eggbox', 'glue', 'iron', 'koalacandy', 'lamp',
               'phone', 'squirrel']

    results = []
    for v in range(0, 1):
        for o in range(6, 7):
            result = track_one_video(root_path, variants[v], objects[o])
            results.append(result)

    # show results
    for r in range(len(results)):
        print(results[r] * 100)



if __name__ == '__main__':
    evaluate_rbot()


