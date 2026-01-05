from __future__ import annotations

import json
import os
import time
import tracemalloc
from statistics import mean

import matplotlib.pyplot as plt

from common.instance import load_instance
from common.pareto import hypervolume_2d, diversity_avg_distance_3d

from exact_bb.solver import solve as solve_exact_bb
from geo_heuristic.solver import solve as solve_geo
from metaheuristic.solver import solve as solve_meta


ALGOS = ["exact_bb", "geo", "meta"]


def objs_3d(solutions):
    return [s.objectives for s in solutions]  # (dist, risk, recharges)


def points_2d_from_objs(objs):
    return [(o[0], o[1]) for o in objs]


def run_one(algo: str, instance_path: str, cfg: dict) -> dict:
    inst = load_instance(instance_path, filter_no_fly_edges=True)

    tracemalloc.start()
    t0 = time.perf_counter()

    if algo == "exact_bb":
        sols = solve_exact_bb(inst, time_limit_s=cfg["time_limit"])
    elif algo == "geo":
        sols = solve_geo(
            inst,
            alphas=cfg["alphas"],
            seed=cfg["seed"],
            runs_per_alpha=cfg["runs_per_alpha"],
            top_k=cfg["top_k"],
        )
    elif algo == "meta":
        sols = solve_meta(
            inst,
            time_limit_s=cfg["time_limit"],
            iters=cfg["iters"],
            seed=cfg["seed"],
            local_swaps=cfg["local_swaps"],
        )
    else:
        raise ValueError("Algoritmo desconocido")

    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    objs = objs_3d(sols)
    pts2d = points_2d_from_objs(objs)

    return {
        "time_s": t1 - t0,
        "mem_peak_bytes": peak,
        "objs": objs,     # guardamos para calcular HV con ref global
        "pts2d": pts2d,   # guardamos para calcular HV con ref global
    }


def plot_curve(out_dir, metric_key, ylabel, title, data, logy=False):
    plt.figure()
    for algo, rows in data.items():
        Ns = [r["N"] for r in rows]
        Ys = [r[metric_key] for r in rows]
        plt.plot(Ns, Ys, marker="o", label=algo)

    if logy:
        plt.yscale("log")

    plt.xlabel("N (destinos)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{metric_key}.png"), dpi=160)


def main():
    data_dir = "data"
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    sizes = [10, 15, 20, 25]
    replicas = 3

    cfg_exact = {
        10: {"time_limit": 8.0},
        15: {"time_limit": 6.0},
        20: {"time_limit": 6.0},
        25: {"time_limit": 6.0},
    }

    cfg_geo = {
        n: {
            "alphas": [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
            "seed": 123,
            "runs_per_alpha": 3,
            "top_k": 3,
        }
        for n in sizes
    }

    cfg_meta = {
        n: {"time_limit": 1.5, "iters": 30_000, "seed": 100 + n, "local_swaps": 2}
        for n in sizes
    }

    # Primero recogemos runs crudos para poder sacar ref global por N
    raw = {a: {n: [] for n in sizes} for a in ALGOS}

    for algo in ALGOS:
        for n in sizes:
            instance_path = os.path.join(data_dir, f"n{n}.json")

            if algo == "exact_bb":
                cfg = cfg_exact[n]
            elif algo == "geo":
                cfg = cfg_geo[n]
            else:
                cfg = cfg_meta[n]

            for _ in range(replicas):
                raw[algo][n].append(run_one(algo, instance_path, cfg))

    # Ref global por N (común a todas las técnicas)
    ref_by_n = {}
    for n in sizes:
        all_pts = []
        for algo in ALGOS:
            for r in raw[algo][n]:
                all_pts.extend(r["pts2d"])
        if all_pts:
            d_ref = max(d for d, _ in all_pts) * 1.1
            r_ref = max(r for _, r in all_pts) * 1.1
        else:
            d_ref, r_ref = 1.0, 1.0
        ref_by_n[n] = (d_ref, r_ref)

    # Ahora construimos summary con HV comparable y diversidad 3D
    summary = {a: [] for a in ALGOS}

    for algo in ALGOS:
        for n in sizes:
            runs = raw[algo][n]
            ref = ref_by_n[n]

            times = [r["time_s"] for r in runs]
            mems = [r["mem_peak_bytes"] for r in runs]
            hvs = []
            divs = []
            nsols = []

            for r in runs:
                objs = r["objs"]
                pts2d = r["pts2d"]
                hv = hypervolume_2d(pts2d, ref=ref)
                div = diversity_avg_distance_3d(objs)
                hvs.append(hv)
                divs.append(div)
                nsols.append(len(objs))

            row = {
                "N": n,
                "time": mean(times),
                "mem_mb": mean(mems) / (1024 * 1024),
                "hv": mean(hvs),
                "div": mean(divs),
                "nsol": mean(nsols),
                "hv_ref": {"d_ref": ref[0], "r_ref": ref[1]},
            }

            summary[algo].append(row)
            print(algo, row)

    with open(os.path.join(out_dir, "summary_all.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_curve(out_dir, "time", "Tiempo medio (s)", "Tiempo vs N", summary, logy=True)
    plot_curve(out_dir, "hv", "Hipervolumen (2D: distancia, riesgo) - ref global", "Hipervolumen vs N", summary)
    plot_curve(out_dir, "div", "Diversidad (3D: distancia, riesgo, recargas)", "Diversidad vs N", summary)

    print("OK: resultados generados en /results")


if __name__ == "__main__":
    main()
