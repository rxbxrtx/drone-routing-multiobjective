from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import random

Obj = Tuple[float, float, int]  # (dist, risk, recharges)


@dataclass
class Solution:
    route: List[int]
    objectives: Obj


def solve(
    instance,
    alphas: List[float] | None = None,
    seed: int = 123,
    runs_per_alpha: int = 3,
    top_k: int = 3,
) -> List[Solution]:
    """
    Heurística geométrica: greedy con aleatoriedad controlada (Top-K) para generar diversidad.
    """
    if alphas is None:
        alphas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    sols: List[Solution] = []
    for a in alphas:
        for r in range(runs_per_alpha):
            rng = random.Random(seed + int(a * 1000) + 31 * r)
            s = _greedy(instance, a, rng, top_k=top_k)
            if s:
                sols.append(s)
    return sols


def _greedy(instance, alpha: float, rng: random.Random, top_k: int) -> Solution | None:
    edges = instance.edges
    hub = instance.hub_id
    deliveries = instance.deliveries
    recharges = instance.recharges
    B = instance.battery_capacity

    visited = set()
    route = [hub]
    current = hub
    battery = B
    dist = risk = 0.0
    k = 0

    stuck_recharges = 0

    while len(visited) < len(deliveries):
        candidates = []
        for d in deliveries:
            if d in visited:
                continue
            w = edges.get((current, d))
            if w and battery >= w[2]:
                score = w[0] + alpha * w[1]
                candidates.append((score, d, w))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            pick = candidates[: max(1, min(top_k, len(candidates)))]
            # elige uno de los top-k (uniforme); mantiene calidad y da diversidad
            _, nxt, w = rng.choice(pick)

            route.append(nxt)
            visited.add(nxt)
            battery -= w[2]
            dist += w[0]
            risk += w[1]
            current = nxt
            stuck_recharges = 0
            continue

        # recargas en cadena
        opts = []
        for r in recharges:
            w = edges.get((current, r))
            if w and battery >= w[2]:
                opts.append((w[0], r, w))

        if not opts:
            return None

        opts.sort(key=lambda x: x[0])
        pick = opts[: max(1, min(2, len(opts)))]  # top-2 recargas
        _, r, w = rng.choice(pick)

        route.append(r)
        dist += w[0]
        risk += w[1]
        k += 1
        current = r
        battery = B

        stuck_recharges += 1
        if stuck_recharges > 8:
            return None

    # cerrar al hub
    w = edges.get((current, hub))
    if w and battery >= w[2]:
        route.append(hub)
        dist += w[0]
        risk += w[1]
        return Solution(route, (dist, risk, k))

    # cerrar vía recargas (máx 3)
    for _ in range(3):
        opts = []
        for r in recharges:
            w1 = edges.get((current, r))
            if w1 and battery >= w1[2]:
                opts.append((w1[0], r, w1))
        if not opts:
            return None

        opts.sort(key=lambda x: x[0])
        pick = opts[: max(1, min(2, len(opts)))]
        _, r, w1 = rng.choice(pick)

        route.append(r)
        dist += w1[0]
        risk += w1[1]
        k += 1
        current = r
        battery = B

        w2 = edges.get((current, hub))
        if w2 and battery >= w2[2]:
            route.append(hub)
            dist += w2[0]
            risk += w2[1]
            return Solution(route, (dist, risk, k))

    return None