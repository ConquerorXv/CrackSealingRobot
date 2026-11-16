from __future__ import annotations

import importlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation

from contracts import Observation
from .data import RGB_EXT, DEPTH_EXT, read_rgb, read_depth


def index_files(directory, extensions):
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f'Missing directory: {root}')
    result = {}
    for path in sorted(root.iterdir()):
        if path.suffix.lower() in extensions:
            if path.stem in result:
                raise ValueError(f'Duplicate input ID: {path.stem}')
            result[path.stem] = path
    return result


def discover(settings):
    if settings.get('rgb_file') or settings.get('depth_file'):
        rgb, depth = Path(settings['rgb_file']), Path(settings['depth_file'])
        if not rgb.is_file() or not depth.is_file():
            raise FileNotFoundError('RGB or depth input file is missing')
        return [(rgb.stem, rgb, depth)]
    root = Path(settings['root'])
    rgb = index_files(root/settings.get('rgb_dir', 'rgb'), RGB_EXT)
    depth = index_files(root/settings.get('depth_dir', 'depth'), DEPTH_EXT)
    if set(rgb) != set(depth):
        raise ValueError(f'Unpaired IDs: {sorted(set(rgb)^set(depth))[:20]}')
    if not rgb:
        raise ValueError('No RGB-D input pairs')
    return [(name, rgb[name], depth[name]) for name in sorted(rgb)]


def canonical_metadata(meta):
    meta = dict(meta)
    if 'ortho_meta' in meta:
        ortho = meta['ortho_meta']
        meta['spacing_m'] = [ortho['grid_x_m_per_pix'], ortho['grid_y_m_per_pix']]
        meta['origin_level_m'] = [ortho['x_min'],
                                  ortho['y_max'] if ortho.get('flip_y') else ortho['y_min']]
        meta['y_sign'] = -1 if ortho.get('flip_y') else 1
        meta['content_shape_hw'] = [ortho['H'], ortho['W']]
    if 'level_transform' in meta:
        transform = np.eye(4)
        transform[:3, :3] = meta['level_transform']['R']
        transform[:3, 3] = meta['level_transform']['t']
        meta['T_level_camera'] = transform.tolist()
    return meta


def prepare_observation(name, rgb, depth_mm, valid=None, metadata=None):
    if rgb.shape[:2] != depth_mm.shape or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RGB and single-channel depth must be spatially aligned')
    meta = canonical_metadata(metadata or {})
    finite = np.isfinite(depth_mm)
    valid = finite if valid is None else np.asarray(valid, bool) & finite
    if valid.shape != depth_mm.shape:
        raise ValueError('Invalid validity map shape')
    depth_mm = np.maximum(np.where(finite, depth_mm, 0), 0).astype(np.float32)
    h, w = depth_mm.shape
    if (h, w) == (736, 1280):
        meta.setdefault('padding_tblr', [8, 8, 0, 0])
    elif h == 720 and w in (1080, 1280):
        rgb = cv2.resize(rgb, (1280, 720), interpolation=cv2.INTER_LINEAR)
        depth_mm = cv2.resize(depth_mm, (1280, 720), interpolation=cv2.INTER_NEAREST)
        valid = cv2.resize(valid.astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST) > 0
        rgb = np.pad(rgb, ((8, 8), (0, 0), (0, 0)))
        depth_mm = np.pad(depth_mm, ((8, 8), (0, 0)))
        valid = np.pad(valid, ((8, 8), (0, 0)))
        meta['padding_tblr'] = [8, 8, 0, 0]
        if 'spacing_m' in meta:
            sx, sy = meta['spacing_m']
            # OpenCV resize maps destination pixel centers into source coordinates.
            ratio = w/1280
            if 'origin_level_m' in meta:
                origin = list(meta['origin_level_m'])
                origin[0] += sx*(.5*ratio-.5)
                meta['origin_level_m'] = origin
            meta['spacing_m'] = [sx*ratio, sy]
        meta['content_shape_hw'] = [720, 1280]
    else:
        raise ValueError(f'Unsupported input shape {h}x{w}; expected 1080x720, 1280x720 or 1280x736')
    # Metadata accompanying an already-resized input must describe the current grid.
    if 'ortho_meta' in meta and (h, w) == (736, 1280):
        original = meta['ortho_meta']
        if original['W'] != 1280:
            sx = float(original['grid_x_m_per_pix'])
            ratio = original['W']/1280
            meta['spacing_m'][0] = sx*ratio
            meta['origin_level_m'][0] += sx*(.5*ratio-.5)
    valid = valid.copy()
    valid[:8] = False
    valid[-8:] = False
    return Observation(name, np.ascontiguousarray(rgb), depth_mm, valid, meta)


def load_observation(name, rgb_path, depth_path, settings):
    metadata = dict(settings.get('metadata') or {})
    meta_path = settings.get('metadata_file')
    if settings.get('metadata_dir'):
        meta_path = str(Path(settings['metadata_dir'])/f'{name}.json')
    if meta_path:
        metadata.update(json.loads(Path(meta_path).read_text(encoding='utf-8')))
    if settings.get('depth_kind', 'depression') != 'depression':
        raise ValueError('Offline depth must be orthographic depression, not camera range')
    unit = settings.get('depth_unit', 'mm')
    if unit not in ('mm', 'm'):
        raise ValueError('Depth unit must be mm or m')
    depth_mm = read_depth(depth_path) * (1000 if unit == 'm' else 1)
    valid = None
    if metadata.get('valid_file'):
        p = Path(metadata['valid_file'])
        if not p.is_absolute() and meta_path:
            p = Path(meta_path).parent/p
        valid = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if valid is None:
            raise FileNotFoundError(p)
        valid = valid > 0
    return prepare_observation(name, read_rgb(rgb_path), depth_mm, valid, metadata)


def fit_plane(points, threshold=.004, iterations=3000, seed=42):
    if len(points) < 100:
        raise ValueError('Insufficient valid depth points for pavement fitting')
    rng = np.random.default_rng(seed)
    points = points[rng.choice(len(points), min(15000, len(points)), replace=False)]
    low, high = np.percentile(points[:, 2], [15, 90])
    candidates = points[(points[:, 2] >= low) & (points[:, 2] <= high)]
    best_mask, best_count = None, -1
    for _ in range(iterations):
        a, b, c = candidates[rng.choice(len(candidates), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal /= length
        offset = -normal@a
        inliers = np.abs(candidates@normal+offset) < threshold
        if inliers.sum() > best_count:
            best_mask, best_count = inliers, int(inliers.sum())
    if best_count < 100:
        raise ValueError('Pavement RANSAC has too few inliers')
    inliers = candidates[best_mask]
    center = inliers.mean(axis=0)
    normal = np.linalg.svd(inliers-center, full_matrices=False)[2][-1]
    if normal[2] < 0:
        normal *= -1
    return normal, float(-normal@center)


def orthographic(rgb, range_m, intrinsics, settings=None):
    settings = settings or {}
    left, right = int(settings.get('crop_left', 100)), int(settings.get('crop_right', 100))
    if left < 0 or right < 0 or left+right >= range_m.shape[1]:
        raise ValueError('Invalid camera crop')
    stop = range_m.shape[1]-right
    rgb, depth = rgb[:, left:stop], range_m[:, left:stop]
    h, w = depth.shape
    if (h, w) not in ((720, 1080), (720, 1280)):
        raise ValueError('Camera crop must yield 1080x720 or 1280x720')
    fx, fy, cx, cy = intrinsics
    cx -= left
    yy, xx = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0) & (depth <= settings.get('max_range_m', 1.))
    z = depth[valid]
    points = np.column_stack(((xx[valid]-cx)*z/fx, (yy[valid]-cy)*z/fy, z))
    colors = rgb[valid]
    normal, offset = fit_plane(points, float(settings.get('plane_threshold_m', .004)),
                                int(settings.get('plane_iterations', 3000)))
    rotation = Rotation.align_vectors([[0., 0., 1.]], [normal])[0].as_matrix()
    translation = np.array([0., 0., offset])
    leveled = points@rotation.T+translation
    corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    rays = np.column_stack(((corners[:, 0]-cx)/fx, (corners[:, 1]-cy)/fy, np.ones(4)))
    denominator = rays@normal
    if (np.abs(denominator) < 1e-8).any() or ((-offset/denominator) <= 0).any():
        raise ValueError('Pavement plane does not intersect camera field of view')
    ground = (rays*(-offset/denominator)[:, None])@rotation.T+translation
    lower, upper = ground[:, :2].min(axis=0), ground[:, :2].max(axis=0)
    spacing = (upper-lower)/np.array([w-1, h-1])
    grid = np.rint((leveled[:, :2]-lower)/spacing).astype(int)
    grid[:, 0] = grid[:, 0].clip(0, w-1)
    grid[:, 1] = grid[:, 1].clip(0, h-1)
    flat = grid[:, 1]*w+grid[:, 0]
    count = np.bincount(flat, minlength=h*w)
    z_max = np.full(h*w, -np.inf)
    np.maximum.at(z_max, flat, leveled[:, 2])
    valid_grid = (count > 0).reshape(h, w)
    depression = np.maximum(np.where(count > 0, z_max, 0), 0).reshape(h, w)
    order = np.lexsort((leveled[:, 2], flat))
    sorted_cells = flat[order]
    first = np.r_[True, np.diff(sorted_cells) != 0]
    ortho_rgb = np.zeros((h*w, 3), dtype=np.uint8)
    ortho_rgb[sorted_cells[first]] = colors[order[first]]
    ortho_rgb = ortho_rgb.reshape(h, w, 3)
    holes = (~valid_grid).astype(np.uint8)*255
    ortho_rgb = cv2.inpaint(ortho_rgb, holes, 3, cv2.INPAINT_TELEA)
    _, nearest = distance_transform_edt(~valid_grid, return_indices=True)
    depression = depression[nearest[0], nearest[1]]
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    metadata = {'spacing_m': spacing.tolist(), 'origin_level_m': lower.tolist(),
                'T_level_camera': transform.tolist(), 'y_sign': 1,
                'content_shape_hw': [h, w], 'depth_definition': 'positive pavement depression',
                'hole_policy': 'nearest depth and inpaint RGB; original validity retained',
                'camera_intrinsics': [fx, fy, cx, cy], 'plane': [*normal.tolist(), offset]}
    # Quantize depression depth to integer millimeters for model input.
    return ortho_rgb, np.floor(depression*1000).astype(np.float32), valid_grid, metadata


def capture_d405(settings, arm=None):
    rs = importlib.import_module('pyrealsense2')
    pipeline = rs.pipeline()
    config = rs.config()
    if settings.get('serial'):
        config.enable_device(settings['serial'])
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    try:
        profile = pipeline.start(config)
        align = rs.align(rs.stream.depth)
        for _ in range(int(settings.get('warmup_frames', 30))):
            frames = pipeline.wait_for_frames(5000)
        frames = align.process(frames)
        d, c = frames.get_depth_frame(), frames.get_color_frame()
        if not d or not c:
            raise RuntimeError('Missing aligned camera frames')
        intr = d.profile.as_video_stream_profile().intrinsics
        if not np.allclose(intr.coeffs, 0, atol=1e-8):
            raise ValueError('Depth intrinsics require undistortion before orthographic reconstruction')
        raw = np.asanyarray(d.get_data()).copy()
        rgb = cv2.cvtColor(np.asanyarray(c.get_data()), cv2.COLOR_BGR2RGB)
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        pose = arm.flange_pose().tolist() if arm else settings.get('T_base_flange')
        ortho_rgb, mm, valid, meta = orthographic(
            rgb, raw.astype(float)*scale, [intr.fx, intr.fy, intr.ppx, intr.ppy], settings)
        meta.update(T_base_flange=pose, camera_depth_scale_m=scale,
                    capture_timestamp_ms=d.get_timestamp())
        observation = prepare_observation('d405_capture', ortho_rgb, mm, valid, meta)
        return observation, {'rgb': rgb, 'depth_raw': raw, 'depth_scale_m': scale}
    finally:
        pipeline.stop()
