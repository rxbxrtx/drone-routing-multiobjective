from __future__ import annotations
from typing import List, Tuple

Point = Tuple[float, float]
Polygon = List[Point]

EPS = 1e-9


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    return (
        min(a[0], b[0]) - EPS <= p[0] <= max(a[0], b[0]) + EPS
        and min(a[1], b[1]) - EPS <= p[1] <= max(a[1], b[1]) + EPS
        and abs(_orient(a, b, p)) <= EPS
    )


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)

    if ((o1 > EPS and o2 < -EPS) or (o1 < -EPS and o2 > EPS)) and (
        (o3 > EPS and o4 < -EPS) or (o3 < -EPS and o4 > EPS)
    ):
        return True

    if abs(o1) <= EPS and _on_segment(p1, p2, q1):
        return True
    if abs(o2) <= EPS and _on_segment(p1, p2, q2):
        return True
    if abs(o3) <= EPS and _on_segment(q1, q2, p1):
        return True
    if abs(o4) <= EPS and _on_segment(q1, q2, p2):
        return True

    return False


def point_in_polygon(pt: Point, poly: Polygon) -> bool:
    x, y = pt
    inside = False
    n = len(poly)

    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]

        if _on_segment(a, b, pt):
            return True

        ax, ay = a
        bx, by = b
        crosses = (ay > y) != (by > y)
        if crosses:
            x_int = ax + (y - ay) * (bx - ax) / (by - ay + 0.0)
            if x_int >= x - EPS:
                inside = not inside

    return inside


def segment_intersects_polygon(a: Point, b: Point, poly: Polygon) -> bool:
    if point_in_polygon(a, poly) or point_in_polygon(b, poly):
        return True

    n = len(poly)
    for i in range(n):
        c = poly[i]
        d = poly[(i + 1) % n]
        if segments_intersect(a, b, c, d):
            return True
    return False


def segment_intersects_any_no_fly(a: Point, b: Point, no_fly: List[Polygon]) -> bool:
    return any(segment_intersects_polygon(a, b, poly) for poly in no_fly)
