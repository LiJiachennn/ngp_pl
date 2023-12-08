import numpy as np
import math
import cv2
from scipy.spatial.transform import Rotation

def compute_error_rotation(R_pred, R_gt):
    trace = np.trace(np.dot(R_pred.T, R_gt))
    trace = min(trace, 3.0)
    return math.acos((trace - 1.0) / 2.0) * 180 / math.pi

def compte_error_translation(t_pred, t_gt):
    return np.linalg.norm(t_pred - t_gt, ord=2)

def compute_error_pose(P_pred, P_gt):
    R_pred = P_pred[0:3, 0:3]
    R_gt = P_gt[0:3, 0:3]
    t_pred = P_pred[0:3, 3]
    t_gt = P_gt[0:3, 3]
    error_rotation = compute_error_rotation(R_pred, R_gt)
    error_translation = compte_error_translation(t_pred, t_gt)
    return error_rotation, error_translation

def compute_success_ncm_ndegree(error_rotation, error_translation, thred_rotation = 5.0, thred_translation = 0.05):
    if error_rotation < thred_rotation and error_translation < thred_translation:
        return 1
    return 0

def load_pose_rbot(path):
    poses = np.loadtxt(path)
    poses_ = []
    for i in range(poses.shape[0]):
        pose_ = np.zeros((4, 4))
        pose_[0:3, 0:3] = poses[i][0:9].reshape(3, 3)
        pose_[0:3, 3] = poses[i][9:12] * 0.001
        pose_[3, :] = [0, 0, 0, 1]
        poses_.append(pose_)
    return np.array(poses_).astype(np.float32)

def depth_normalize(depth_, depth_min):
    depth = depth_.copy()
    depth[depth < depth_min] = depth_min
    depth = (depth - depth_min) / (depth.max() - depth_min)
    depth = (depth*255).astype(np.uint8)
    return depth

def depth2pseudo(depth_, depth_min):
    depth = depth_.copy()
    depth = depth_normalize(depth, depth_min)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    return depth

def eulerT2RT(euler_angles, t):
    R = euler2R(euler_angles)
    RT = np.eye(4)
    RT[:3, :3] = R
    RT[:3, 3] = t
    return RT

def RT2eulerT(RT):
    euler = R2euler(RT[:3, :3])
    t = RT[:3, 3]
    eulerT = np.concatenate((euler, t))
    return eulerT

def euler2R(euler_angles):
    r = Rotation.from_euler('xyz', euler_angles, degrees=True)
    R = r.as_matrix()
    return R

def R2euler(R):
    r = Rotation.from_matrix(R)
    euler_angles = r.as_euler('xyz', degrees=True)
    return euler_angles