from __future__ import annotations

import cv2
import numpy as np

from .kinematics import transform_matrix


def calibrate_intrinsics(object_points, image_points, image_size_wh):
    if len(object_points) != len(image_points) or len(image_points) < 3:
        raise ValueError('Intrinsic calibration needs at least three paired board views')
    objects = [np.asarray(p, np.float32) for p in object_points]
    images = [np.asarray(p, np.float32) for p in image_points]
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        objects, images, tuple(image_size_wh), None, None)
    return {'reprojection_rms_px': float(rms), 'camera_matrix': matrix,
            'distortion': distortion, 'rvecs': rvecs, 'tvecs': tvecs}


def calibrate_handeye(T_base_flange_views, T_camera_target_views):
    if len(T_base_flange_views) != len(T_camera_target_views) or len(T_base_flange_views) < 3:
        raise ValueError('Hand-eye calibration needs at least three paired, diverse poses')
    flange = [transform_matrix(t, 'T_base_flange') for t in T_base_flange_views]
    target = [transform_matrix(t, 'T_camera_target') for t in T_camera_target_views]
    rotation, translation = cv2.calibrateHandEye(
        [t[:3, :3] for t in flange], [t[:3, 3] for t in flange],
        [t[:3, :3] for t in target], [t[:3, 3] for t in target],
        method=cv2.CALIB_HAND_EYE_PARK)
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, translation.reshape(3)
    return transform_matrix(result, 'T_flange_camera')


def calibrate_tcp_pivot(T_base_flange_views):
    if len(T_base_flange_views) < 3:
        raise ValueError('TCP pivot calibration needs at least three diverse flange poses')
    transforms = [transform_matrix(t) for t in T_base_flange_views]
    design = np.concatenate([np.column_stack((t[:3, :3], -np.eye(3))) for t in transforms])
    rhs = np.concatenate([-t[:3, 3] for t in transforms])
    solution, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
    if rank < 6:
        raise ValueError('TCP calibration poses do not constrain a unique pivot')
    residuals = (design@solution-rhs).reshape(-1, 3)
    transform = np.eye(4)
    transform[:3, 3] = solution[:3]
    return {'T_flange_nozzle': transform, 'pivot_base_m': solution[3:],
            'rms_m': float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))}
