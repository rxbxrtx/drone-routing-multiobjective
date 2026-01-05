from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import time

Obj = Tuple[float, float, int]  # (dist, risk, recharges)


@dataclass(frozen=True)
class Solution:
    route: List[int]
    objectives: Obj


def dominates_or_equal(a: Obj, b: Obj) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2]


def pareto_add(front: List[Solution], cand: Solution) -> bool:
    c = cand.objectives
    for s in front:
        if dominates_or_equal(s.objectives, c):
            return False
    new_front = []
    for s in front:
        if not dominates_or_equal(c, s.objectives):
            new_front.append(s)
    new_front.append(cand)
    front[:] = new_front
    return True


def solve(
    instance,
    time_limit_s: float = 2.0,
    iters: int = 30_000,
    seed: int = 123,
    local_swaps: int = 2,
) -> List[Solution]:
    rng = random.Random(seed)
    deliveries = list(instance.deliveries)
    front: List[Solution] = []

    t0 = time.perf_counter()
    i = 0
    while i < iters and (time.perf_counter() - t0) < time_limit_s:
        rng.shuffle(deliveries)
        base = _build(instance, deliveries)
        if base:
            pareto_add(front, base)

            perm = deliveries[:]
            for _ in range(local_swaps):
                a, b = rng.randrange(len(perm)), rng.randrange(len(perm))
                perm[a], perm[b] = perm[b], perm[a]
                s2 = _build(instance, perm)
                if s2:
                    pareto_add(front, s2)
        i += 1

    return front


def _build(instance, perm: List[int]) -> Optional[Solution]:
    edges = instance.edges
    hub = instance.hub_id
    recharges = instance.recharges
    B = instance.battery_capacity

    route = [hub]
    current = hub
    battery = B
    dist = risk = 0.0
    k = 0

    for nxt in perm:
        # Intento directo
        w = edges.get((current, nxt))
        if w and battery >= w[2]:
            route.append(nxt)
            battery -= w[2]
            dist += w[0]
            risk += w[1]
            current = nxt
            continue

        # Si no, permito cadena de recargas para poder alcanzar nxt
        success = False
        for _step in range(6):
            best = None
            for r in recharges:
                w1 = edges.get((current, r))
                w2 = edges.get((r, nxt))
                if not w1 or not w2:
                    continue
                if battery < w1[2] or B < w2[2]:
                    continue
                extra = w1[0] + w2[0]
                if best is None or extra < best[0]:
                    best = (extra, r, w1, w2)

            if best is not None:
                _, r, w1, w2 = best
                # ir a recarga
                route.append(r)
                dist += w1[0]
                risk += w1[1]
                k += 1
                current = r
                battery = B
                # ir al nxt
                route.append(nxt)
                dist += w2[0]
                risk += w2[1]
                battery = B - w2[2]
                current = nxt
                success = True
                break

            # No hay recarga que conecte con nxt: me muevo a la recarga alcanzable más cercana
            opts = []
            for r in recharges:
                w1 = edges.get((current, r))
                if w1 and battery >= w1[2]:
                    opts.append((r, w1))
            if not opts:
                return None

            r, w1 = min(opts, key=lambda x: x[1][0])
            route.append(r)
            dist += w1[0]
            risk += w1[1]
            k += 1
            current = r
            battery = B

        if not success:
            return None

    # Cerrar al hub, permitiendo cadena corta
    w = edges.get((current, hub))
    if w and battery >= w[2]:
        route.append(hub)
        dist += w[0]
        risk += w[1]
        return Solution(route, (dist, risk, k))

    for _ in range(3):
        opts = []
        for r in recharges:
            w1 = edges.get((current, r))
            if w1 and battery >= w1[2]:
                opts.append((r, w1))
        if not opts:
            return None

        r, w1 = min(opts, key=lambda x: x[1][0])
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

