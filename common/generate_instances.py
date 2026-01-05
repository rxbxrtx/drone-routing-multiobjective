from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from common.geometry import segment_intersects_any_no_fly, Polygon

Point = Tuple[float, float]


@dataclass
class GenConfig:
    width: float = 100.0
    height: float = 100.0

    battery_capacity: float = 350.0
    num_recharges: int = 3

    num_no_fly: int = 3
    rect_min_size: float = 12.0
    rect_max_size: float = 22.0

    battery_per_distance: float = 0.55
    battery_noise: float = 0.06
    risk_base: float = 0.2
    risk_near_poly_boost: float = 2.2

    max_tries: int = 200
    min_edge_ratio: float = 0.80


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def rect_polygon(cx: float, cy: float, w: float, h: float) -> Polygon:
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def poly_center(poly: Polygon) -> Point:
    x = sum(p[0] for p in poly) / len(poly)
    y = sum(p[1] for p in poly) / len(poly)
    return (x, y)


def risk_for_edge(a: Point, b: Point, no_fly: List[Polygon], dist: float, cfg: GenConfig) -> float:
    risk = cfg.risk_base * dist
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    for poly in no_fly:
        cx, cy = poly_center(poly)
        d = math.hypot(mx - cx, my - cy)
        risk += cfg.risk_near_poly_boost * (1.0 / (1.0 + 0.15 * d))
    return risk


def sample_points(rng: random.Random, n: int, cfg: GenConfig, no_fly: List[Polygon]) -> List[Point]:
    pts: List[Point] = []
    tries = 0
    while len(pts) < n and tries < cfg.max_tries * 50:
        tries += 1
        x = rng.uniform(0, cfg.width)
        y = rng.uniform(0, cfg.height)
        if segment_intersects_any_no_fly((x, y), (x, y), no_fly):
            continue
        pts.append((x, y))
    if len(pts) < n:
        raise RuntimeError("No se pudieron muestrear puntos válidos fuera de no-fly.")
    return pts


def build_edges(
    coords: Dict[int, Point],
    all_nodes: List[int],
    no_fly: List[Polygon],
    cfg: GenConfig,
    rng: random.Random,
) -> List[dict]:
    edges_json: List[dict] = []
    for u in all_nodes:
        for v in all_nodes:
            if u == v:
                continue
            a, b = coords[u], coords[v]
            if no_fly and segment_intersects_any_no_fly(a, b, no_fly):
                continue

            dist = euclidean(a, b)
            risk = risk_for_edge(a, b, no_fly, dist, cfg)

            noise = rng.uniform(1.0 - cfg.battery_noise, 1.0 + cfg.battery_noise)
            battery = dist * cfg.battery_per_distance * noise

            edges_json.append({"from": u, "to": v, "distance": dist, "risk": risk, "battery": battery})
    return edges_json


def edge_ratio(num_edges: int, n_nodes: int) -> float:
    full = n_nodes * (n_nodes - 1)
    return num_edges / full if full > 0 else 0.0


def generate_instance(n_deliveries: int, seed: int, cfg: GenConfig) -> dict:
    rng = random.Random(seed)

    for _ in range(cfg.max_tries):
        no_fly: List[Polygon] = []
        for _ in range(cfg.num_no_fly):
            w = rng.uniform(cfg.rect_min_size, cfg.rect_max_size)
            h = rng.uniform(cfg.rect_min_size, cfg.rect_max_size)
            cx = rng.uniform(w / 2, cfg.width - w / 2)
            cy = rng.uniform(h / 2, cfg.height - h / 2)
            no_fly.append(rect_polygon(cx, cy, w, h))

        total_pts = 1 + n_deliveries + cfg.num_recharges
        pts = sample_points(rng, total_pts, cfg, no_fly)

        hub_id = 0
        deliveries_ids = list(range(1, 1 + n_deliveries))
        recharges_ids = list(range(100, 100 + cfg.num_recharges))

        coords: Dict[int, Point] = {hub_id: pts[0]}
        for i, nid in enumerate(deliveries_ids, start=1):
            coords[nid] = pts[i]
        for j, nid in enumerate(recharges_ids, start=1 + n_deliveries):
            coords[nid] = pts[j]

        all_nodes = [hub_id] + deliveries_ids + recharges_ids
        edges = build_edges(coords, all_nodes, no_fly, cfg, rng)

        if edge_ratio(len(edges), len(all_nodes)) < cfg.min_edge_ratio:
            continue

        return {
            "battery_capacity": cfg.battery_capacity,
            "hub": {"id": hub_id, "x": coords[hub_id][0], "y": coords[hub_id][1]},
            "deliveries": [{"id": nid, "x": coords[nid][0], "y": coords[nid][1]} for nid in deliveries_ids],
            "recharges": [{"id": nid, "x": coords[nid][0], "y": coords[nid][1]} for nid in recharges_ids],
            "edges": edges,
            "no_fly_zones": [{"polygon": [[x, y] for (x, y) in poly]} for poly in no_fly],
            "meta": {"seed": seed, "n_deliveries": n_deliveries, "num_recharges": cfg.num_recharges, "num_no_fly": cfg.num_no_fly},
        }

    raise RuntimeError("No se pudo generar una instancia válida.")


def write_instances(out_dir: str = "data") -> None:
    cfg = GenConfig()
    sizes = [10, 15, 20, 25]
    base_seed = 12345

    import os
    os.makedirs(out_dir, exist_ok=True)

    for n in sizes:
        if n == 15:
            inst = None
            for k in range(50):
                seed = base_seed + n * 100 + 7
                try:
                    inst = generate_instance(n_deliveries=n, seed=seed, cfg=cfg)
                    break
                except RuntimeError:
                    continue
            if inst is None:
                raise RuntimeError("No se pudo generar una instancia válida para N=15 tras 50 semillas.")
        else:
            seed = base_seed + n * 100
            inst = generate_instance(n_deliveries=n, seed=seed, cfg=cfg)

        path = os.path.join(out_dir, f"n{n}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(inst, f, ensure_ascii=False, indent=2)
        print(f"Wrote {path}")



if __name__ == "__main__":
    write_instances()
