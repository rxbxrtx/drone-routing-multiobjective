from __future__ import annotations
from typing import List, Tuple
import math

Obj = Tuple[float, float, int]  # (dist, risk, recharges)


def nondominated_2d(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Filtra no dominadas en 2D (min)."""
    pts = sorted(points, key=lambda x: (x[0], x[1]))
    nd = []
    best_r = float("inf")
    for d, r in pts:
        if r < best_r:
            nd.append((d, r))
            best_r = r
    return nd


def hypervolume_2d(points: List[Tuple[float, float]], ref: Tuple[float, float]) -> float:
    """
    Hipervolumen dominado (minimización) en 2D respecto a ref=(D_ref, R_ref).
    """
    nd = nondominated_2d(points)
    if not nd:
        return 0.0

    nd.sort(key=lambda x: x[0])

    hv = 0.0
    prev_d = ref[0]

    for d, r in reversed(nd):
        width = prev_d - d
        height = ref[1] - r
        if width > 0 and height > 0:
            hv += width * height
        prev_d = d

    return hv


def diversity_avg_distance_3d(objs: List[Obj]) -> float:
    """
    Diversidad 3D: distancia media entre soluciones en el espacio (dist, risk, recargas).
    No exige no-dominancia; mide dispersión del set de soluciones.
    """
    if len(objs) < 2:
        return 0.0

    pts = sorted(objs, key=lambda x: (x[0], x[1], x[2]))
    total = 0.0
    for i in range(len(pts) - 1):
        d1, r1, k1 = pts[i]
        d2, r2, k2 = pts[i + 1]
        total += math.sqrt((d2 - d1) ** 2 + (r2 - r1) ** 2 + (k2 - k1) ** 2)

    return total / (len(pts) - 1)


