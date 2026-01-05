from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

from common.geometry import segment_intersects_any_no_fly, Polygon

EdgeW = Tuple[float, float, float]  # (distance, risk, battery)


@dataclass
class Instance:
    battery_capacity: float
    hub_id: int
    deliveries: List[int]
    recharges: List[int]
    coords: Dict[int, Tuple[float, float]]
    edges: Dict[Tuple[int, int], EdgeW]
    no_fly: List[Polygon]


def load_instance(path: str, filter_no_fly_edges: bool = True) -> Instance:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    B = float(data["battery_capacity"])

    hub = data["hub"]
    hub_id = int(hub["id"])

    coords: Dict[int, Tuple[float, float]] = {}
    coords[hub_id] = (float(hub["x"]), float(hub["y"]))

    deliveries = [int(v["id"]) for v in data.get("deliveries", [])]
    for v in data.get("deliveries", []):
        coords[int(v["id"])] = (float(v["x"]), float(v["y"]))

    recharges = [int(v["id"]) for v in data.get("recharges", [])]
    for v in data.get("recharges", []):
        coords[int(v["id"])] = (float(v["x"]), float(v["y"]))

    no_fly: List[Polygon] = []
    for z in data.get("no_fly_zones", []):
        poly = [(float(x), float(y)) for x, y in z["polygon"]]
        if len(poly) >= 3:
            no_fly.append(poly)

    edges: Dict[Tuple[int, int], EdgeW] = {}
    for e in data.get("edges", []):
        u = int(e["from"])
        v = int(e["to"])
        d = float(e["distance"])
        r = float(e["risk"])
        b = float(e["battery"])

        if filter_no_fly_edges and (u in coords and v in coords) and no_fly:
            a = coords[u]
            c = coords[v]
            if segment_intersects_any_no_fly(a, c, no_fly):
                continue

        edges[(u, v)] = (d, r, b)

    return Instance(
        battery_capacity=B,
        hub_id=hub_id,
        deliveries=deliveries,
        recharges=recharges,
        coords=coords,
        edges=edges,
        no_fly=no_fly,
    )
