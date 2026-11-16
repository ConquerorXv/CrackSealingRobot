from __future__ import annotations

import argparse
import csv
from datetime import datetime
import importlib
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from contracts import write_json
from perception.inference import Predictor, overlay
from perception.inputs import discover, load_observation, capture_d405
from quantification.geometry import quantify
from trajectory_planning.planner import plan, render_route
from control.adapters import SimulatedArm, SimulatedPump, UnitreeArm, ModbusPump
from control.execution import preflight, prepare_mission, execute


PROJECT = Path(__file__).resolve().parent


def read_config(path=PROJECT/'config.yaml'):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    def resolve(value):
        if value is None:
            return None
        p = Path(value)
        return str((path.parent/p).resolve()) if not p.is_absolute() else str(p)
    for section, keys in [('input', ['root', 'rgb_file', 'depth_file', 'metadata_dir', 'metadata_file']),
                          ('perception', ['weights']), ('output', ['root'])]:
        for key in keys:
            if key in cfg.get(section, {}):
                cfg[section][key] = resolve(cfg[section][key])
    return cfg


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Cannot write image: {path}')


def export_quantification(directory, result):
    write_json(directory/'quantification.json', {
        'physical_available': result.physical_available, 'missing': result.missing,
        'features': result.features, 'segments': result.segments,
        'volume_definition': 'sum of section sweeps; junction overlap is not union-corrected'})
    with (directory/'segments.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['segment_id', 'closed', 'length_px', 'mean_width_px',
                         'length_m', 'mean_width_mm', 'max_depth_mm', 'mean_area_mm2',
                         'volume_ml', 'physical_status'])
        for segment in result.segments:
            g = segment.geometry
            writer.writerow([segment.id, segment.closed, g['length_px'], float(np.mean(g['width_px'])),
                             g['length_m'], None if g['width_m'] is None else float(np.mean(g['width_m']))*1000,
                             float(np.max(g['max_depth_mm'])),
                             None if g['area_m2'] is None else float(np.mean(g['area_m2']))*1e6,
                             g['volume_ml'], g['physical_status']])


def run(cfg):
    source = cfg['input']['source']
    mode = cfg['control']['mode']
    if source not in ('offline', 'd405') or mode not in ('plan', 'simulate', 'real'):
        raise ValueError('input.source must be offline/d405 and control.mode plan/simulate/real')
    if mode != 'plan':
        preflight(cfg['control'])
    if source == 'd405':
        if importlib.util.find_spec('pyrealsense2') is None:
            raise RuntimeError('D405 input requires pyrealsense2')
        if mode == 'real' and not cfg['control'].get('capture_pose_verified'):
            raise ValueError('Set capture_pose_verified only after placing the arm at a stable observation pose')
    inputs = discover(cfg['input']) if source == 'offline' else None
    if mode == 'real' and source == 'offline' and len(inputs) != 1:
        raise ValueError('Real execution requires one spatially calibrated observation per mission')
    predictor = Predictor(cfg['perception']['weights'], cfg['perception'].get('device', 'auto'),
                          cfg['perception'].get('threshold', .5))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = Path(cfg['output']['root'])/f'run_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'configuration.json', cfg)
    arm = pump = None
    records = []
    try:
        if source == 'd405':
            if mode == 'real':
                arm = UnitreeArm(cfg['control'])
            observation, raw = capture_d405(cfg['input'].get('camera', {}), arm)
            np.savez_compressed(output/'camera_raw.npz', **raw)
            write_json(output/'camera_metadata.json', observation.metadata)
            write_image(output/'camera_ortho_rgb.png', cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR))
            np.save(output/'camera_ortho_depth_mm.npy', observation.depth_mm)
            write_image(output/'camera_valid.png', observation.valid.astype(np.uint8)*255)
            inputs = [(observation.name, None, None)]
        for index, (name, rgb_path, depth_path) in enumerate(inputs):
            directory = output/name
            directory.mkdir()
            started = time.perf_counter()
            record = {'id': name}
            events = []
            try:
                if source == 'offline':
                    observation = load_observation(name, rgb_path, depth_path, cfg['input'])
                write_json(directory/'metadata.json', observation.metadata)
                inference_start = time.perf_counter()
                mask = predictor.predict(observation)
                record['prediction_seconds'] = time.perf_counter()-inference_start
                write_image(directory/'mask.png', mask)
                write_image(directory/'overlay.png', overlay(observation.rgb, mask,
                                                             cfg['perception'].get('overlay_alpha', .5)))
                quant_start = time.perf_counter()
                quantified = quantify(mask, observation.depth_mm, observation.valid,
                                      observation.metadata, cfg.get('quantification'))
                record['quantification_seconds'] = time.perf_counter()-quant_start
                write_image(directory/'skeleton.png', quantified.skeleton.astype(np.uint8)*255)
                export_quantification(directory, quantified)
                planned = plan(quantified.segments, observation.metadata, cfg.get('planning'))
                write_json(directory/'route.json', planned)
                write_image(directory/'route.png', render_route(observation.rgb, quantified.segments, planned))
                record.update(segments=len(quantified.segments), planning_seconds=planned['seconds'],
                              physical_available=quantified.physical_available, units=planned['units'],
                              transfer_length=planned['transfer_length'], missing=quantified.missing)
                if not quantified.segments:
                    record['status'] = 'no_task'
                elif mode == 'plan':
                    record['status'] = 'planned'
                    record['execution'] = 'not_requested'
                    if not quantified.physical_available:
                        record['execution_unavailable_reason'] = 'Missing metric scale or invalid cross sections'
                else:
                    preflight(cfg['control'], observation.metadata)
                    if arm is None:
                        arm = SimulatedArm() if mode == 'simulate' else UnitreeArm(cfg['control'])
                    plans = prepare_mission(arm, quantified.segments, planned,
                                            observation.metadata, cfg['control'])
                    # No pump starts until the entire mission has valid joint trajectories.
                    if pump is None:
                        pump = SimulatedPump() if mode == 'simulate' else ModbusPump(cfg['control']['pump'])
                    execute(arm, pump, plans, cfg['control'], events)
                    record.update(status='simulated' if mode == 'simulate' else 'executed',
                                  pump_rate_modes=sorted({p['rate_mode'] for p in plans}))
            except BaseException as error:
                record.update(status='failed', error=f'{type(error).__name__}: {error}')
                if mode == 'real' or not isinstance(error, Exception):
                    raise
            finally:
                record['total_seconds'] = time.perf_counter()-started
                write_json(directory/'execution.json', {'mode': mode, 'events': events})
                write_json(directory/'result.json', record)
                records.append(record)
                write_json(output/'summary.json', {'samples': len(records), 'mode': mode,
                           'failed': sum(r['status']=='failed' for r in records), 'records': records})
            print(f"[{index+1}/{len(inputs)}] {name}: {record['status']}"
                  f" ({record['total_seconds']:.2f}s)", flush=True)
    finally:
        try:
            if pump is not None:
                try:
                    pump.stop()
                finally:
                    pump.close()
        finally:
            if arm is not None:
                arm.close()
    print(f'Results: {output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT/'config.yaml')
    args = parser.parse_args()
    output = run(read_config(args.config))
    import json
    summary = json.loads((output/'summary.json').read_text())
    if summary['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
