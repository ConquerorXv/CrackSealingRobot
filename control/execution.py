from __future__ import annotations

import importlib.util
from pathlib import Path
import time

import numpy as np

from .kinematics import map_surface_points, smooth_path, joint_path, transform_matrix


def preflight(settings, metadata=None):
    missing = []
    real = settings.get('mode') == 'real'
    required = ['T_flange_camera', 'T_flange_nozzle', 'nozzle_rpy', 'safe_z_m',
                'work_clearance_m', 'workspace_min_m', 'workspace_max_m']
    for key in required:
        if settings.get(key) is None:
            missing.append(f'control.{key}')
    pump = settings.get('pump', {})
    for key in ['reverse_rpm', 'reverse_seconds', 'max_rpm']:
        if pump.get(key) is None:
            missing.append(f'control.pump.{key}')
    if pump.get('flow_m3_s_per_rpm') is None and pump.get('fixed_rpm') is None:
        missing.append('control.pump.flow_m3_s_per_rpm or fixed_rpm')
    if real:
        for key in ['sdk_path', 'sdk_dt_s', 'calibration_verified', 'jacobian_verified']:
            if not settings.get(key):
                missing.append(f'control.{key}')
        if not Path(settings.get('sdk_path') or '/nonexistent').is_dir():
            missing.append('Unitree SDK directory')
        if importlib.util.find_spec('pymodbus') is None:
            missing.append('pymodbus==3.6.9')
        for key in ['port', 'slave', 'forward_value', 'reverse_value', 'stop_value',
                    'speed_register', 'operation_register', 'register_units_per_rpm']:
            if pump.get(key) is None:
                missing.append(f'control.pump.{key}')
    if metadata is not None:
        for key in ['spacing_m', 'origin_level_m', 'T_level_camera', 'T_base_flange']:
            if metadata.get(key) is None:
                missing.append(f'observation.metadata.{key}')
    if missing:
        raise ValueError('Missing execution requirements: '+', '.join(missing))
    for key in ['T_flange_camera', 'T_flange_nozzle']:
        transform_matrix(settings[key], key)
    if pump['max_rpm'] <= 0 or pump['reverse_rpm'] < 0 or pump['reverse_seconds'] < 0:
        raise ValueError('Invalid pump speed or reversal duration')
    if pump['reverse_rpm'] > pump['max_rpm']:
        raise ValueError('Reverse speed exceeds max_rpm')
    if pump.get('flow_m3_s_per_rpm') is not None and pump['flow_m3_s_per_rpm'] <= 0:
        raise ValueError('Pump flow coefficient must be positive')
    if pump.get('fixed_rpm') is not None and not 0 < pump['fixed_rpm'] <= pump['max_rpm']:
        raise ValueError('Fixed pump speed is outside the allowed range')
    if real:
        forward, reverse, stop = (pump[k] for k in ['forward_value', 'reverse_value', 'stop_value'])
        if len({forward, reverse, stop}) != 3 or any(not 0 <= v <= 65535 for v in [forward, reverse, stop]):
            raise ValueError('Pump direction and stop values must be distinct 16-bit values')
        if pump['register_units_per_rpm'] <= 0:
            raise ValueError('Pump register conversion must be positive')
    if np.any(np.asarray(settings['workspace_min_m']) >= settings['workspace_max_m']):
        raise ValueError('Invalid workspace bounds')
    if settings['mode'] not in ('simulate', 'real'):
        raise ValueError('Execution mode must be simulate or real')


def segment_rate(segment, settings):
    pump = settings['pump']
    speed = float(settings.get('speed_m_s', .06))
    if speed <= 0:
        raise ValueError('Sealing speed must be positive')
    coefficient = pump.get('flow_m3_s_per_rpm')
    if coefficient is None:
        return speed, float(pump['fixed_rpm']), 'manual_fixed_rpm'
    density = segment.geometry.get('mean_linear_density_m2')
    if density is None:
        raise ValueError('Adaptive pump rate requires valid metric volume quantification')
    if density <= 0:
        raise ValueError('Zero-volume segment cannot request extrusion')
    rpm = density*speed/coefficient
    if rpm > pump['max_rpm']:
        speed *= pump['max_rpm']/rpm
        rpm = float(pump['max_rpm'])
    return speed, rpm, 'segment_volume_adaptive'


def validate_path(points, settings):
    lower, upper = np.asarray(settings['workspace_min_m']), np.asarray(settings['workspace_max_m'])
    if not np.isfinite(points).all() or (points < lower).any() or (points > upper).any():
        raise ValueError('Trajectory exceeds the configured Cartesian workspace')


def prepare_mission(arm, segments, route, metadata, settings):
    preflight(settings, metadata)
    by_id = {s.id: s for s in segments}
    frequency = float(settings.get('frequency_hz', 50))
    if frequency <= 0:
        raise ValueError('Trajectory frequency must be positive')
    q = arm.current_q()
    T_flange_nozzle = transform_matrix(settings['T_flange_nozzle'])
    current_tcp = (arm.fk(q)@T_flange_nozzle)[:3, 3]
    plans = []
    for item in route['route']:
        segment = by_id[item['segment_id']]
        if len(segment.points_yx) < 2:
            raise ValueError('An isolated skeleton point cannot be executed')
        pixels = segment.points_yx[::-1] if item['reverse'] else segment.points_yx
        points = map_surface_points(pixels, metadata, settings['T_flange_camera'],
                                    float(settings['work_clearance_m']))
        speed, rpm, rate_mode = segment_rate(segment, settings)
        seal = smooth_path(points, speed, frequency)
        validate_path(seal, settings)
        if settings['safe_z_m'] < max(seal[:, 2].max(), current_tcp[2]):
            raise ValueError('safe_z_m must be above the current TCP and working path')
        lift = current_tcp.copy(); lift[2] = settings['safe_z_m']
        above = seal[0].copy(); above[2] = settings['safe_z_m']
        transfer = []
        for a, b in [(current_tcp, lift), (lift, above), (above, seal[0])]:
            if np.linalg.norm(a-b) > 1e-9:
                transfer.append(smooth_path([a, b], float(settings.get('transfer_speed_m_s', .08)), frequency))
        approach = np.concatenate(transfer) if transfer else seal[:1]
        end_above = seal[-1].copy(); end_above[2] = settings['safe_z_m']
        retract = smooth_path([seal[-1], end_above], float(settings.get('transfer_speed_m_s', .08)), frequency)
        joints = []
        for path in [approach, seal, retract]:
            validate_path(path, settings)
            joint = joint_path(arm, path, settings['nozzle_rpy'], q, T_flange_nozzle, settings)
            # Include the first transition from the preceding command in the limit check.
            velocities = np.diff(np.vstack((q, joint)), axis=0)*frequency
            if (np.abs(velocities) > arm.speed_limits()).any():
                raise ValueError('Joint trajectory exceeds SDK velocity limits')
            if np.max(np.abs(np.diff(np.vstack((q, joint)), axis=0))) > settings.get('max_joint_step_rad', .15):
                raise ValueError('Discontinuous joint trajectory')
            joints.append(joint)
            q = joint[-1]
        plans.append({'segment_id': segment.id, 'approach': joints[0], 'seal': joints[1],
                      'retract': joints[2], 'rpm': rpm, 'speed_m_s': speed, 'rate_mode': rate_mode})
        current_tcp = end_above
    return plans


def stream(arm, joints, settings):
    frequency = float(settings.get('frequency_hz', 50))
    real = settings['mode'] == 'real'
    period = float(settings['sdk_dt_s']) if real else 1/frequency
    if period <= 0 or period > 1/frequency:
        raise ValueError('SDK communication period must be positive and no longer than a trajectory interval')
    duration = (len(joints)-1)/frequency
    times = np.arange(0, duration+period*.5, period)
    times = np.r_[times[times < duration], duration]
    sample_t = np.arange(len(joints))/frequency
    interpolated = np.column_stack([np.interp(times, sample_t, joints[:, i]) for i in range(6)])
    qd = np.gradient(interpolated, times, axis=0) if len(times) > 1 else np.zeros_like(interpolated)
    if len(qd):
        qd[0] = 0.
        qd[-1] = 0.
    start = time.monotonic()
    for index, (q, velocity) in enumerate(zip(interpolated, qd)):
        if real:
            delay = start+times[index]-time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif -delay > settings.get('max_lateness_s', .1):
                raise TimeoutError('Servo communication missed its deadline')
        arm.send(q, velocity)
        if real and np.max(np.abs(arm.current_q()-q)) > settings.get('max_tracking_error_rad', .15):
            raise RuntimeError('Measured joint tracking error exceeds the configured limit')


    if len(interpolated):
        arm.send(interpolated[-1], np.zeros(6))
    if real and len(interpolated):
        target_q = interpolated[-1]
        target_tcp = arm.fk(target_q) @ transform_matrix(settings['T_flange_nozzle'])
        deadline = time.monotonic()+float(settings.get('arrival_timeout_s', 2.))
        from scipy.spatial.transform import Rotation
        while True:
            actual = arm.fk(arm.current_q()) @ transform_matrix(settings['T_flange_nozzle'])
            position_error = np.linalg.norm(actual[:3, 3]-target_tcp[:3, 3])
            rotation_error = np.linalg.norm(Rotation.from_matrix(
                target_tcp[:3, :3]@actual[:3, :3].T).as_rotvec())
            if (position_error <= settings.get('arrival_position_tolerance_m', .001)
                    and rotation_error <= settings.get('arrival_rotation_tolerance_rad', .01)):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('TCP did not reach the trajectory endpoint')
            arm.send(target_q, np.zeros(6))
            time.sleep(period)


def execute(arm, pump, plans, settings, events=None):
    events = [] if events is None else events
    started = time.monotonic()

    def record(event, segment=None, **fields):
        events.append({'seconds': time.monotonic()-started, 'event': event,
                       'segment_id': segment, **fields})

    try:
        pump.stop()
        for plan in plans:
            sid = plan['segment_id']
            record('transfer_and_approach', sid)
            stream(arm, plan['approach'], settings)
            pump.set_speed(plan['rpm'])
            pump.forward()
            record('extrusion', sid, rpm=plan['rpm'], speed_m_s=plan['speed_m_s'], rate_mode=plan['rate_mode'])
            stream(arm, plan['seal'], settings)
            pump.stop()
            pump.set_speed(settings['pump']['reverse_rpm'])
            pump.reverse()
            record('reverse', sid, rpm=settings['pump']['reverse_rpm'],
                   duration_s=settings['pump']['reverse_seconds'])
            if settings['mode'] == 'real':
                time.sleep(settings['pump']['reverse_seconds'])
            pump.stop()
            record('pump_stopped', sid)
            stream(arm, plan['retract'], settings)
            record('retracted', sid)
    except BaseException as error:
        record('aborted', error=str(error))
        try:
            pump.stop()
        finally:
            try:
                arm.send(arm.current_q(), np.zeros(6))
            except Exception as hold_error:
                record('hold_failed', error=str(hold_error))
        raise
    finally:
        try:
            pump.stop()
        finally:
            record('final_stop_attempted')
    return events
