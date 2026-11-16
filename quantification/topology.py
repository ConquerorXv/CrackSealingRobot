from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from contracts import Segment


OFFSETS = [(0, 1), (-1, 1), (-1, 0), (-1, -1),
           (0, -1), (1, -1), (1, 0), (1, 1)]


def crossing_numbers(skeleton):
    padded = np.pad(skeleton.astype(np.int8), 1)
    h, w = skeleton.shape
    neighbors = np.stack([padded[1+dy:1+dy+h, 1+dx:1+dx+w]
                          for dy, dx in OFFSETS], axis=-1)
    return np.abs(neighbors - np.roll(neighbors, -1, axis=-1)).sum(-1) // 2


def skeleton_graph(skeleton):
    points = {tuple(p) for p in np.argwhere(skeleton)}
    graph = {}
    for y, x in sorted(points):
        neighbors = []
        for dy, dx in OFFSETS:
            other = (y + dy, x + dx)
            if other not in points:
                continue
            # Remove redundant diagonal shortcuts around a connected pixel corner.
            if dy and dx and ((y + dy, x) in points or (y, x + dx) in points):
                continue
            neighbors.append(other)
        graph[(y, x)] = sorted(neighbors)
    return graph


def edge_key(a, b):
    return (a, b) if a < b else (b, a)


def prune_skeleton(skeleton, min_component_length_px=20., min_branch_length_px=20.):
    """Remove short acyclic components and terminal branches, preserving cycles."""
    limits = np.asarray([min_component_length_px, min_branch_length_px], dtype=float)
    if not np.isfinite(limits).all() or (limits < 0).any():
        raise ValueError('Skeleton pruning lengths must be finite and nonnegative')
    component_limit, branch_limit = limits
    skeleton = np.asarray(skeleton, dtype=bool).copy()
    stats = {'input_pixels': int(skeleton.sum()), 'removed_pixels': 0,
             'removed_components': 0, 'removed_branches': 0, 'iterations': 0,
             'min_component_length_px': float(component_limit),
             'min_branch_length_px': float(branch_limit)}
    while True:
        graph = skeleton_graph(skeleton)
        remove = set()
        remaining = set(graph)
        while remaining and component_limit > 0:
            component, pending = set(), [min(remaining)]
            while pending:
                point = pending.pop()
                if point in component:
                    continue
                component.add(point)
                pending.extend(p for p in graph[point] if p not in component)
            remaining.difference_update(component)
            edges = [(p, q) for p in component for q in graph[p] if p < q]
            length = sum(np.hypot(p[0]-q[0], p[1]-q[1]) for p, q in edges)
            if len(edges) < len(component) and length < component_limit:
                remove.update(component)
                stats['removed_components'] += 1
        if branch_limit > 0:
            for start, neighbors in graph.items():
                if len(neighbors) != 1 or start in remove:
                    continue
                path = [start]
                previous, current = start, neighbors[0]
                length = float(np.hypot(current[0]-previous[0], current[1]-previous[1]))
                while len(graph[current]) == 2:
                    path.append(current)
                    next_point = next(p for p in graph[current] if p != previous)
                    length += np.hypot(next_point[0]-current[0], next_point[1]-current[1])
                    previous, current = current, next_point
                if len(graph[current]) >= 3 and length < branch_limit:
                    remove.update(path)
                    stats['removed_branches'] += 1
        if not remove:
            break
        y, x = np.asarray(sorted(remove)).T
        skeleton[y, x] = False
        stats['iterations'] += 1
    stats['removed_pixels'] = stats['input_pixels'] - int(skeleton.sum())
    return skeleton, stats


def parse_topology(mask, min_component_length_px=20., min_branch_length_px=20.):
    skeleton = skeletonize(np.asarray(mask) > 0, method='lee').astype(bool)
    skeleton, pruning = prune_skeleton(skeleton, min_component_length_px,
                                      min_branch_length_px)
    crossing = crossing_numbers(skeleton)
    graph = skeleton_graph(skeleton)
    nodes = {p for p, neighbors in graph.items()
             if len(neighbors) != 2 or crossing[p] >= 3}
    visited, segments = set(), []

    def trace(start, neighbor):
        path = [start, neighbor]
        visited.add(edge_key(start, neighbor))
        previous, current = start, neighbor
        while current not in nodes and current != start:
            choices = [p for p in graph[current]
                       if p != previous and edge_key(current, p) not in visited]
            if not choices:
                break
            next_point = choices[0]
            visited.add(edge_key(current, next_point))
            path.append(next_point)
            previous, current = current, next_point
        segments.append(Segment(len(segments), np.asarray(path, dtype=np.int32),
                                path[0] == path[-1]))

    for start in sorted(nodes):
        if not graph[start]:
            segments.append(Segment(len(segments), np.asarray([start], dtype=np.int32)))
        for neighbor in graph[start]:
            if edge_key(start, neighbor) not in visited:
                trace(start, neighbor)

    # Remaining edges form disconnected cycles; choose the closest feature node.
    feature_tree = cKDTree(sorted(nodes)) if nodes else None
    remaining = {p for p, neighbors in graph.items()
                 if any(edge_key(p, q) not in visited for q in neighbors)}
    while remaining:
        root = min(remaining)
        component, pending = set(), [root]
        while pending:
            p = pending.pop()
            if p in component:
                continue
            component.add(p)
            pending.extend(q for q in graph[p]
                           if edge_key(p, q) not in visited and q not in component)
        candidates = sorted(component)
        start = candidates[0]
        if feature_tree is not None:
            distances = feature_tree.query(np.asarray(candidates))[0]
            start = candidates[int(np.argmin(distances))]
        for neighbor in graph[start]:
            if edge_key(start, neighbor) not in visited:
                trace(start, neighbor)
        remaining = {p for p in remaining
                     if any(edge_key(p, q) not in visited for q in graph[p])}

    features = {'endpoints_yx': np.argwhere(skeleton & (crossing == 1)),
                'intersections_yx': np.argwhere(skeleton & (crossing >= 3)),
                'isolated_yx': np.argwhere(skeleton & (crossing == 0)),
                'graph_edges': sum(map(len, graph.values())) // 2,
                'traced_edges': len(visited), 'pruning': pruning}
    return skeleton, segments, features
