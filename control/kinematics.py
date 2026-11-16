from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


def transform_matrix(value, name='transform'):
    matrix = np.asarray(value, float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f'{name} must be a finite 4x4 matrix')
    rotation = matrix[:3, :3]
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f'{name} has an invalid homogeneous row')
    if not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError(f'{name} rotation is not SO(3)')
    return matrix


def map_surface_points(points_yx, metadata, T_flange_camera, clearance_m=0.):
    required = ['spacing_m', 'origin_level_m', 'T_level_camera', 'T_base_flange']
    missing = [key for key in required if metadata.get(key) is None]
    if missing:
        raise ValueError(f'Missing spatial metadata: {missing}')
    spacing, origin = np.asarray(metadata['spacing_m']), np.asarray(metadata['origin_level_m'])
    top, _, left, _ = metadata.get('padding_tblr', [8, 8, 0, 0])
    pixels = np.asarray(points_yx, float)-[top, left]
    level = np.column_stack((origin[0]+pixels[:, 1]*spacing[0],
                             origin[1]+pixels[:, 0]*spacing[1]*metadata.get('y_sign', 1),
                             np.zeros(len(pixels)), np.ones(len(pixels))))
    T_base_level = (transform_matrix(metadata['T_base_flange'], 'T_base_flange')
                    @ transform_matrix(T_flange_camera, 'T_flange_camera')
                    @ np.linalg.inv(transform_matrix(metadata['T_level_camera'], 'T_level_camera')))
    points = (T_base_level@level.T).T[:, :3]
    points[:, 2] += clearance_m
    return points


def smooth_path(points, speed_m_s, frequency_hz=50):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Path must be a finite Nx3 array')
    if speed_m_s <= 0 or frequency_hz <= 0:
        raise ValueError('Speed and frequency must be positive')
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]
    points = points[keep]
    if len(points) < 2:
        raise ValueError('Segment has no executable length')
    cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    spline = CubicSpline(cumulative, points, axis=0)
    dense_u = np.linspace(0, cumulative[-1], max(100, len(points)*4))
    dense = spline(dense_u)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    duration = arc[-1]/speed_m_s
    steps = max(2, int(np.ceil(duration*frequency_hz))+1)
    target_arc = np.linspace(0, arc[-1], steps)
    sampled_u = np.interp(target_arc, arc, dense_u)
    return spline(sampled_u)


def dls_ik(model, target, seed, damping=.02, iterations=150,
           position_tolerance=1e-4, rotation_tolerance=1e-3):
    target = transform_matrix(target, 'IK target')
    q = np.asarray(seed, float).copy()
    lower, upper = model.joint_limits()
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('IK seed must contain six finite joint positions')
    for _ in range(iterations):
        current = model.fk(q)
        dp = target[:3, 3]-current[:3, 3]
        dr = Rotation.from_matrix(target[:3, :3]@current[:3, :3].T).as_rotvec()
        if np.linalg.norm(dp) <= position_tolerance and np.linalg.norm(dr) <= rotation_tolerance:
            return q
        error = np.r_[dp, dr]
        jacobian = np.asarray(model.jacobian(q), float)
        if jacobian.shape != (6, 6) or not np.isfinite(jacobian).all():
            raise ValueError('Invalid geometric Jacobian (linear rows, then angular rows)')
        step = jacobian.T@np.linalg.solve(jacobian@jacobian.T+damping**2*np.eye(6), error)
        largest = np.max(np.abs(step))
        if largest > .15:
            step *= .15/largest
        q = np.clip(q+step, lower, upper)
    raise RuntimeError('DLS inverse kinematics failed to converge')


def joint_path(model, points, rpy, seed, T_flange_nozzle, settings):
    rotation = Rotation.from_euler('xyz', rpy).as_matrix()
    T_nozzle_flange = np.linalg.inv(transform_matrix(T_flange_nozzle, 'T_flange_nozzle'))
    joints, q = [], np.asarray(seed, float)
    for point in points:
        target = np.eye(4)
        target[:3, :3], target[:3, 3] = rotation, point
        q = dls_ik(model, target@T_nozzle_flange,
                   q, damping=float(settings.get('dls_damping', .02)))
        joints.append(q.copy())
    return np.asarray(joints)
