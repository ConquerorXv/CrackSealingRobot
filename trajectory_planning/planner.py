from __future__ import annotations

import time
import cv2
import numpy as np


def route_cost(route, distances):
    return float(sum(distances[2*a+(1-da), 2*b+db]
                     for (a, da), (b, db) in zip(route[:-1], route[1:])))


def two_opt(route, distances):
    route = list(route)
    n = len(route)
    if n < 2:
        return route
    while True:
        best_gain, best_swap = -1e-10, None
        for i in range(n):
            for j in range(i, n):
                a, da = route[i]
                b, db = route[j]
                gain = 0.
                if i:
                    p, dp = route[i-1]
                    gain += distances[2*p+1-dp, 2*b+1-db] - distances[2*p+1-dp, 2*a+da]
                if j+1 < n:
                    q, dq = route[j+1]
                    gain += distances[2*a+da, 2*q+dq] - distances[2*b+1-db, 2*q+dq]
                if gain < best_gain:
                    best_gain, best_swap = gain, (i, j)
        if best_swap is None:
            return route
        i, j = best_swap
        route[i:j+1] = [(s, 1-d) for s, d in reversed(route[i:j+1])]


def plan(segments, metadata=None, settings=None):
    settings, metadata = settings or {}, metadata or {}
    started = time.perf_counter()
    n = len(segments)
    spacing = metadata.get('spacing_m')
    scale = np.asarray(spacing[::-1] if spacing is not None else [1., 1.])
    if not n:
        return {'route': [], 'units': 'm' if spacing else 'px', 'transfer_length': 0.,
                'working_length': 0., 'history': [], 'seconds': time.perf_counter()-started}
    endpoints = np.asarray([[s.points_yx[0], s.points_yx[-1]] for s in segments], float)
    flat = (endpoints * scale).reshape(-1, 2)
    distances = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    alpha, beta, rho = (float(settings.get(k, v)) for k, v in
                        [('alpha', 2.52), ('beta', 3.35), ('rho', .04)])
    ants, iterations = int(settings.get('ants', 10)), int(settings.get('iterations', 100))
    tau_min, tau_max = float(settings.get('tau_min', .01)), float(settings.get('tau_max', 1.))
    if not (alpha >= 0 and beta >= 0 and 0 < rho < 1 and ants > 0 and iterations > 0
            and 0 < tau_min <= tau_max):
        raise ValueError('Invalid MMAS parameters')
    rng = np.random.default_rng(int(settings.get('seed', 42)))
    tau = np.full_like(distances, tau_max)
    best, best_cost, history = None, float('inf'), []
    # A deterministic feasible incumbent bounds every subsequent stochastic result.
    for direction in (0, 1):
        route, unvisited = [(0, direction)], set(range(1, n))
        while unvisited:
            a, da = route[-1]
            candidate = min(((s, d) for s in sorted(unvisited) for d in (0, 1)),
                            key=lambda pair: distances[2*a+1-da, 2*pair[0]+pair[1]])
            route.append(candidate)
            unvisited.remove(candidate[0])
        route = two_opt(route, distances)
        cost = route_cost(route, distances)
        if cost < best_cost:
            best, best_cost = route, cost
    if n > 1:
        for _ in range(iterations):
            iteration_route, iteration_cost = None, float('inf')
            for _ in range(ants):
                start = int(rng.integers(n))
                route = [(start, int(rng.integers(2)))]
                available = np.ones(n, bool)
                available[start] = False
                while available.any():
                    candidates = np.flatnonzero(available)
                    entries = (2*candidates[:, None]+np.array([0, 1])).ravel()
                    a, da = route[-1]
                    exit_index = 2*a+1-da
                    log_prob = alpha*np.log(tau[exit_index, entries]) - beta*np.log(
                        np.maximum(distances[exit_index, entries], 1e-12))
                    prob = np.exp(log_prob-log_prob.max())
                    prob /= prob.sum()
                    entry = int(rng.choice(entries, p=prob))
                    route.append((entry//2, entry % 2))
                    available[entry//2] = False
                cost = route_cost(route, distances)
                if cost < iteration_cost:
                    iteration_route, iteration_cost = route, cost
            candidate = two_opt(iteration_route, distances)
            cost = route_cost(candidate, distances)
            if cost < best_cost:
                best, best_cost = candidate, cost
            tau *= 1-rho
            deposit = 1/max(best_cost, 1e-12)
            for (a, da), (b, db) in zip(best[:-1], best[1:]):
                i, j = 2*a+1-da, 2*b+db
                tau[i, j] += deposit
                tau[j, i] += deposit
            np.clip(tau, tau_min, tau_max, out=tau)
            history.append(best_cost)
    working = sum(float(np.linalg.norm(np.diff(s.points_yx, axis=0)*scale, axis=1).sum())
                  for s in segments)
    return {'route': [{'segment_id': segments[s].id, 'reverse': bool(d)} for s, d in best],
            'units': 'm' if spacing is not None else 'px', 'transfer_length': best_cost,
            'working_length': working, 'history': history,
            'parameters': {'alpha': alpha, 'beta': beta, 'rho': rho, 'ants': ants,
                           'iterations': iterations, 'seed': settings.get('seed', 42)},
            'seconds': time.perf_counter()-started}


def render_route(rgb, segments, result):
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    indexed = {s.id: s for s in segments}
    last = None
    for index, item in enumerate(result['route']):
        points = indexed[item['segment_id']].points_yx
        points = points[::-1] if item['reverse'] else points
        xy = np.ascontiguousarray(points[:, ::-1], dtype=np.int32)
        if last is not None:
            cv2.line(image, tuple(last), tuple(xy[0]), (180, 180, 180), 1)
        color = tuple(int(v) for v in np.random.default_rng(index).integers(50, 255, 3))
        cv2.polylines(image, [xy], False, color, 2)
        cv2.putText(image, str(index+1), tuple(xy[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (0, 255, 255), 1, cv2.LINE_AA)
        last = xy[-1]
    return image
