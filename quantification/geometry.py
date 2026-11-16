from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.integrate import trapezoid

from contracts import Quantification
from .topology import parse_topology


def normals_for_segment(points, closed=False, neighbors=11, spacing_yx=None):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.tile([0., 1.], (len(points), 1))
    radius = max(1, neighbors // 2)
    indices = np.arange(len(points))[:, None] + np.arange(-radius, radius+1)
    if closed:
        indices %= max(1, len(points)-1)
    else:
        indices = indices.clip(0, len(points)-1)
    scale = np.ones(2) if spacing_yx is None else np.asarray(spacing_yx)
    local = points[indices] * scale
    centered = local - local.mean(axis=1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', centered, centered)
    _, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0] / scale
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def sample(array, points, order=1):
    return map_coordinates(np.asarray(array, dtype=float), np.asarray(points).T,
                           order=order, mode='constant', cval=0., prefilter=False)


def ray_distances(mask, centers, normals, sign, step=.25, max_distance=256.):
    active = np.ones(len(centers), dtype=bool)
    inside = np.zeros(len(centers))
    outside = np.zeros(len(centers))
    for distance in np.arange(step, max_distance+step, step):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        positions = centers[indices] + sign * distance * normals[indices]
        hit = sample(mask, positions) < .5
        outside[indices[hit]] = distance
        inside[indices[~hit]] = distance
        active[indices[hit]] = False
    # Refine the binary-mask 0.5 isocontour to subpixel precision.
    for _ in range(6):
        mid = (inside + outside) / 2
        values = sample(mask, centers + sign * mid[:, None] * normals)
        hit = values < .5
        outside = np.where(hit & ~active, mid, outside)
        inside = np.where(~hit & ~active, mid, inside)
    return (inside + outside) / 2, ~active


def quantify(mask, depth_mm, valid=None, metadata=None, settings=None):
    settings, metadata = settings or {}, metadata or {}
    mask = np.asarray(mask) > 0
    valid = np.ones_like(mask) if valid is None else np.asarray(valid, bool)
    if depth_mm.shape != mask.shape or valid.shape != mask.shape:
        raise ValueError('Mask, depth and valid map must share the same shape')
    finite = np.isfinite(depth_mm)
    valid = valid & finite
    depth_m = np.maximum(np.where(finite, depth_mm, 0), 0) * .001
    skeleton, segments, features = parse_topology(
        mask, settings.get('min_component_length_px', 20.),
        settings.get('min_branch_length_px', 20.))
    spacing = metadata.get('spacing_m')
    if spacing is not None:
        spacing = np.asarray(spacing, float)
        if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError('spacing_m must contain positive finite X/Y meters per pixel')
    missing = [] if spacing is not None else ['Missing metric X/Y ground sampling distance (spacing_m)']
    for segment in segments:
        points = segment.points_yx.astype(float)
        normals = normals_for_segment(points, segment.closed, settings.get('pca_neighbors', 11),
                                      None if spacing is None else spacing[::-1])
        step = float(settings.get('ray_step_px', .25))
        limit = float(settings.get('max_ray_px', 256))
        if step <= 0 or limit <= 0:
            raise ValueError('Ray step and distance must be positive')
        left, left_ok = ray_distances(mask, points, normals, -1, step, limit)
        right, right_ok = ray_distances(mask, points, normals, 1, step, limit)
        widths = left + right
        n_profile = max(2, int(np.ceil(widths.max(initial=0) / step))+1)
        relative = -left[:, None] + widths[:, None] * np.linspace(0, 1, n_profile)
        positions = points[:, None, :] + relative[:, :, None] * normals[:, None, :]
        profiles = sample(depth_m, positions.reshape(-1, 2)).reshape(len(points), n_profile)
        support = sample(valid, positions.reshape(-1, 2), order=0).reshape(len(points), n_profile) > .5
        section_ok = left_ok & right_ok & support.all(axis=1)
        lengths_px = np.linalg.norm(np.diff(points, axis=0), axis=1)
        geometry = {'length_px': float(lengths_px.sum()), 'width_px': widths,
                    'normals_yx': normals, 'boundaries_yx': np.stack([
                        points-left[:, None]*normals, points+right[:, None]*normals], axis=1),
                    'cross_sections_valid': section_ok,
                    'max_depth_mm': profiles.max(axis=1) * 1000,
                    'length_m': None, 'width_m': None, 'area_m2': None,
                    'volume_m3': None, 'volume_ml': None, 'mean_linear_density_m2': None,
                    'physical_status': 'missing_scale' if spacing is None else 'invalid_depth_or_ray'}
        if spacing is not None and section_ok.all():
            # Y/X vectors use the corresponding metric spacing, including anisotropy.
            metric_step = np.linalg.norm(normals * spacing[::-1], axis=1)
            section_width = widths * metric_step
            area = trapezoid(profiles, x=relative * metric_step[:, None], axis=1)
            edge_lengths = np.linalg.norm(np.diff(points, axis=0) * spacing[::-1], axis=1)
            length = float(edge_lengths.sum())
            volume = float(np.sum((area[:-1]+area[1:])*.5*edge_lengths))
            geometry.update(length_m=length, width_m=section_width, area_m2=area,
                            volume_m3=volume, volume_ml=volume*1e6,
                            mean_linear_density_m2=volume/length if length > 0 else 0.,
                            physical_status='available')
        segment.geometry = geometry
    return Quantification(skeleton, segments, features,
                          spacing is not None and all(s.geometry['physical_status']=='available' for s in segments),
                          missing)
