from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
import math

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


class ExactBBSolver:
    def __init__(
        self,
        edges: Dict[Tuple[int, int], Tuple[float, float, float]],
        hub_id: int,
        deliveries: List[int],
        recharges: List[int],
        battery_capacity: float,
    ):
        self.edges = edges
        self.hub = hub_id
        self.deliveries = deliveries
        self.recharges = recharges
        self.B = battery_capacity

        self.idx = {nid: i for i, nid in enumerate(deliveries)}
        self.full_mask = (1 << len(deliveries)) - 1

        self.front: List[Solution] = []
        self.best_state: Dict[Tuple[int, int], Obj] = {}

        self.start_t = 0.0
        self.time_limit = 0.0

    def solve(self, time_limit_s: float) -> List[Solution]:
        self.start_t = time.perf_counter()
        self.time_limit = time_limit_s
        self.front.clear()
        self.best_state.clear()

        self._dfs(self.hub, 0, self.B, 0.0, 0.0, 0, [self.hub])
        return self.front

    def _dfs(
        self,
        current: int,
        mask: int,
        battery: float,
        dist: float,
        risk: float,
        k: int,
        route: List[int],
    ):
        if time.perf_counter() - self.start_t > self.time_limit:
            return

        key = (current, mask)
        cur = (dist, risk, k)
        if key in self.best_state and dominates_or_equal(self.best_state[key], cur):
            return
        self.best_state[key] = cur

        if mask == self.full_mask:
            self._close(current, battery, dist, risk, k, route)
            return

        for d in self.deliveries:
            bit = 1 << self.idx[d]
            if mask & bit:
                continue

            w = self.edges.get((current, d))
            if w and battery >= w[2]:
                self._dfs(
                    d,
                    mask | bit,
                    battery - w[2],
                    dist + w[0],
                    risk + w[1],
                    k,
                    route + [d],
                )
                continue

            for r in self.recharges:
                w1 = self.edges.get((current, r))
                w2 = self.edges.get((r, d))
                if not w1 or not w2:
                    continue
                if battery < w1[2] or self.B < w2[2]:
                    continue
                self._dfs(
                    d,
                    mask | bit,
                    self.B - w2[2],
                    dist + w1[0] + w2[0],
                    risk + w1[1] + w2[1],
                    k + 1,
                    route + [r, d],
                )

    def _close(self, current, battery, dist, risk, k, route):
        w = self.edges.get((current, self.hub))
        if w and battery >= w[2]:
            pareto_add(self.front, Solution(route + [self.hub], (dist + w[0], risk + w[1], k)))
            return

        for r in self.recharges:
            w1 = self.edges.get((current, r))
            w2 = self.edges.get((r, self.hub))
            if not w1 or not w2:
                continue
            if battery < w1[2] or self.B < w2[2]:
                continue
            pareto_add(
                self.front,
                Solution(
                    route + [r, self.hub],
                    (dist + w1[0] + w2[0], risk + w1[1] + w2[1], k + 1),
                ),
            )


def solve(instance, time_limit_s: float = 15.0, node_limit: int = 0) -> List[Solution]:
    solver = ExactBBSolver(
        edges=instance.edges,
        hub_id=instance.hub_id,
        deliveries=instance.deliveries,
        recharges=instance.recharges,
        battery_capacity=instance.battery_capacity,
    )
    return solver.solve(time_limit_s)
