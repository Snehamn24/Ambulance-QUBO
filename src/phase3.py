"""Phase III: dynamic fleet re-dispatch with adaptive QUBO penalty tuning.

Phase I optimizes one route. Phase II assigns multiple available ambulances to
simultaneous emergencies and hospital-capacity slots. Phase III adds a
rolling-horizon experiment layer in which the same fleet problem is re-solved
under changing traffic snapshots (normal, peak, accident and roadblock), and an
adaptive penalty tuner selects a QUBO hard-constraint penalty from a small,
pre-declared candidate set using raw simulated-annealing feasibility.

This remains an academic simulation. Traffic, emergency arrivals, hospital
queues/capacities and deadlines are synthetic. A decision epoch is a fresh
re-optimization snapshot; continuous vehicle motion and future traffic are not
predicted. No quantum advantage is claimed.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import csv
import json
import math
import statistics

import matplotlib.pyplot as plt
import networkx as nx
from dwave.samplers import SimulatedAnnealingSampler

from src.data import load_graph
from src.routes import baseline
from src.traffic import snapshot
from src.phase2 import (
    _largest_scc_patch,
    assignment_metrics,
    build_dispatch_qubo,
    build_options,
    exact_dispatch,
    make_entities,
    plot_dispatch,
    solve_dispatch,
    validate_assignment,
)


PHASE3_METHODS = ("exact_dynamic", "fixed_hybrid", "adaptive_hybrid")
DEFAULT_PENALTY_FACTORS = (1.05, 1.25, 1.5, 2.0, 3.0, 4.0)
DEFAULT_TARGET_FEASIBILITY = 0.60


def _write_json(payload, path):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(rows, path):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _raw_feasible_fraction(samples, options, emergencies):
    feasible_reads = 0
    total_reads = 0
    for record in samples.data(["sample", "num_occurrences"]):
        occurrences = int(record.num_occurrences)
        total_reads += occurrences
        selected = [variable for variable, bit in record.sample.items() if bit]
        if validate_assignment(selected, options, emergencies) is not None:
            feasible_reads += occurrences
    return feasible_reads / total_reads if total_reads else 0.0


def choose_penalty_from_trace(trace, target_feasibility=DEFAULT_TARGET_FEASIBILITY):
    """Choose the smallest tested penalty that reaches the target.

    If no tested factor reaches the target, choose the factor with the greatest
    observed feasibility, breaking ties toward the smaller penalty. This helper
    is deterministic and separately unit-testable.
    """
    if not trace:
        raise ValueError("Penalty trace cannot be empty")
    if not 0 <= target_feasibility <= 1:
        raise ValueError("target_feasibility must be between 0 and 1")
    cleaned = []
    for row in trace:
        factor = float(row["penalty_factor"])
        feasible = float(row["raw_feasible_fraction"])
        if not math.isfinite(factor) or factor <= 1:
            raise ValueError("Penalty factors must be finite and exceed 1")
        if not math.isfinite(feasible) or not 0 <= feasible <= 1:
            raise ValueError("Feasibility fractions must be in [0, 1]")
        cleaned.append((factor, feasible))
    reached = sorted((factor, feasible) for factor, feasible in cleaned if feasible >= target_feasibility)
    if reached:
        return reached[0][0]
    best_feasibility = max(feasible for _, feasible in cleaned)
    return min(factor for factor, feasible in cleaned if abs(feasible - best_feasibility) <= 1e-12)


def probe_penalties(
    options,
    emergencies,
    seed=0,
    penalty_factors=DEFAULT_PENALTY_FACTORS,
    target_feasibility=DEFAULT_TARGET_FEASIBILITY,
    reads=20,
    sweeps=250,
):
    """Estimate raw-SA feasibility for candidate QUBO penalty factors."""
    if reads < 1 or sweeps < 1:
        raise ValueError("reads and sweeps must be positive")
    factors = tuple(float(value) for value in penalty_factors)
    if not factors or any((not math.isfinite(value) or value <= 1) for value in factors):
        raise ValueError("penalty_factors must contain finite values greater than 1")
    if len(set(factors)) != len(factors):
        raise ValueError("penalty_factors must be unique")

    sampler = SimulatedAnnealingSampler()
    trace = []
    for factor in sorted(factors):
        bqm, info = build_dispatch_qubo(options, emergencies, penalty_factor=factor)
        samples = sampler.sample(
            bqm,
            num_reads=reads,
            num_sweeps=sweeps,
            seed=seed,
        )
        trace.append({
            "penalty_factor": factor,
            "qubo_penalty": info["penalty"],
            "raw_feasible_fraction": _raw_feasible_fraction(samples, options, emergencies),
        })
    selected = choose_penalty_from_trace(trace, target_feasibility)
    return selected, trace


def solve_adaptive_hybrid(
    options,
    emergencies,
    seed=0,
    reads=64,
    sweeps=1000,
    penalty_factors=DEFAULT_PENALTY_FACTORS,
    target_feasibility=DEFAULT_TARGET_FEASIBILITY,
    probe_reads=20,
    probe_sweeps=250,
):
    """Tune the QUBO penalty from raw-SA feasibility, then run hybrid dispatch."""
    started = perf_counter()
    selected_factor, trace = probe_penalties(
        options,
        emergencies,
        seed=seed,
        penalty_factors=penalty_factors,
        target_feasibility=target_feasibility,
        reads=probe_reads,
        sweeps=probe_sweeps,
    )
    result = solve_dispatch(
        options,
        emergencies,
        method="hybrid_dispatch",
        seed=seed,
        reads=reads,
        sweeps=sweeps,
        penalty_factor=selected_factor,
    )
    result["method"] = "adaptive_hybrid"
    result["selected_penalty_factor"] = selected_factor
    result["target_feasibility"] = target_feasibility
    result["penalty_trace"] = trace
    result["solve_runtime_ms"] = result["runtime_ms"]
    result["runtime_ms"] = (perf_counter() - started) * 1000
    return result


def solve_fixed_hybrid(options, emergencies, seed=0, reads=64, sweeps=1000, penalty_factor=2.0):
    result = solve_dispatch(
        options,
        emergencies,
        method="hybrid_dispatch",
        seed=seed,
        reads=reads,
        sweeps=sweeps,
        penalty_factor=penalty_factor,
    )
    result["method"] = "fixed_hybrid"
    result["selected_penalty_factor"] = float(penalty_factor)
    result["target_feasibility"] = None
    result["penalty_trace"] = []
    return result


def _event_edge(base_graph, source, target):
    normal = snapshot(base_graph, 12, "normal")
    route = baseline(normal, source, target, "dijkstra")
    return route[len(route) // 2] if route else None


def _required_pairs(ambulances, emergencies, hospitals):
    available_ambulances = [a for a in ambulances if a.get("available", True)]
    available_hospitals = [h for h in hospitals if h.get("available", True) and int(h.get("capacity", 0)) > 0]
    pairs = []
    for ambulance in available_ambulances:
        for emergency in emergencies:
            pairs.append((ambulance["node"], emergency["node"]))
    for emergency in emergencies:
        for hospital in available_hospitals:
            pairs.append((emergency["node"], hospital["node"]))
    return pairs


def _safe_roadblock_edge(base_graph, ambulances, emergencies, hospitals):
    """Pick an on-route edge whose removal preserves required reachability."""
    normal = snapshot(base_graph, 12, "normal")
    candidates = []
    for ambulance in [a for a in ambulances if a.get("available", True)]:
        for emergency in emergencies:
            try:
                route = baseline(normal, ambulance["node"], emergency["node"], "dijkstra")
            except nx.NetworkXNoPath:
                continue
            candidates.extend(route)
    seen = set()
    candidates = [edge for edge in candidates if not (edge in seen or seen.add(edge))]
    for edge in candidates:
        blocked = snapshot(base_graph, 8, "roadblock", [edge])
        if all(nx.has_path(blocked, source, target) for source, target in _required_pairs(ambulances, emergencies, hospitals)):
            return edge
    # Fallback to any removable edge preserving the same required reachability.
    for _, _, edge in base_graph.edges(keys=True):
        if edge in seen:
            continue
        blocked = snapshot(base_graph, 8, "roadblock", [edge])
        if all(nx.has_path(blocked, source, target) for source, target in _required_pairs(ambulances, emergencies, hospitals)):
            return edge
    return None


def build_dynamic_scenario(root, emergency_count=3):
    full = load_graph(Path(root) / "data" / "road_graph.json")
    base = _largest_scc_patch(full, 60)
    ambulances, emergencies, hospitals = make_entities(base, emergency_count)
    primary_source = next(a["node"] for a in ambulances if a.get("available", True))
    primary_target = max(emergencies, key=lambda e: (int(e["severity"]), e["id"]))["node"]
    incident_edge = _event_edge(base, primary_source, primary_target)
    roadblock_edge = _safe_roadblock_edge(base, ambulances, emergencies, hospitals)
    events = [
        {"epoch": 0, "scenario": "normal", "hour": 12, "event_edges": []},
        {"epoch": 1, "scenario": "peak", "hour": 8, "event_edges": []},
        {"epoch": 2, "scenario": "accident", "hour": 8, "event_edges": [incident_edge] if incident_edge else []},
        {"epoch": 3, "scenario": "roadblock", "hour": 8, "event_edges": [roadblock_edge] if roadblock_edge else []},
    ]
    return base, ambulances, emergencies, hospitals, events


def _run_method(options, emergencies, method, seed, reads, sweeps, target_feasibility):
    if method == "exact_dynamic":
        result = exact_dispatch(options, emergencies)
        result["method"] = "exact_dynamic"
        result["selected_penalty_factor"] = None
        result["penalty_trace"] = []
        return result
    if method == "fixed_hybrid":
        return solve_fixed_hybrid(options, emergencies, seed, reads, sweeps, penalty_factor=2.0)
    if method == "adaptive_hybrid":
        return solve_adaptive_hybrid(
            options,
            emergencies,
            seed=seed,
            reads=reads,
            sweeps=sweeps,
            target_feasibility=target_feasibility,
            probe_reads=max(8, min(20, reads)),
            probe_sweeps=max(100, min(250, sweeps)),
        )
    raise ValueError("Unknown Phase III method")


def _flatten(result, emergency_count, event, seed, exact_score):
    gap = None
    if result.get("status") == "ok" and result.get("objective_score_s") is not None:
        gap = max(0.0, 100.0 * (result["objective_score_s"] - exact_score) / exact_score)
    return {
        "emergency_count": emergency_count,
        "epoch": event["epoch"],
        "scenario": event["scenario"],
        "hour": event["hour"],
        "event_edge_count": len(event["event_edges"]),
        "seed": seed,
        "method": result["method"],
        "status": result["status"],
        "objective_score_s": result.get("objective_score_s"),
        "gap_pct": gap,
        "mean_response_time_s": result.get("mean_response_time_s"),
        "max_response_time_s": result.get("max_response_time_s"),
        "deadline_misses": result.get("deadline_misses"),
        "runtime_ms": result.get("runtime_ms"),
        "raw_feasible_fraction": result.get("raw_feasible_fraction"),
        "candidate_retained": result.get("candidate_retained", False),
        "improved_over_candidate": result.get("improved_over_candidate", False),
        "selected_penalty_factor": result.get("selected_penalty_factor"),
        "qubo_variables": result.get("qubo_variables", 0),
        "qubo_interactions": result.get("qubo_interactions", 0),
    }


def _plot_runtime(rows, path):
    sizes = sorted({row["emergency_count"] for row in rows})
    plt.figure(figsize=(8, 5))
    for method in PHASE3_METHODS:
        values = []
        for size in sizes:
            subset = [row["runtime_ms"] for row in rows if row["emergency_count"] == size and row["method"] == method]
            values.append(statistics.median(subset) if subset else float("nan"))
        plt.plot(sizes, values, marker="o", label=method)
    plt.xlabel("Simultaneous emergencies")
    plt.ylabel("Median runtime (ms)")
    plt.title("Phase III dynamic dispatch runtime scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_penalties(rows, path):
    adaptive = [row for row in rows if row["method"] == "adaptive_hybrid" and row["selected_penalty_factor"] is not None]
    scenarios = [scenario for scenario in ("normal", "peak", "accident", "roadblock") if any(row["scenario"] == scenario for row in adaptive)]
    plt.figure(figsize=(8, 5))
    values = []
    for scenario in scenarios:
        subset = [row["selected_penalty_factor"] for row in adaptive if row["scenario"] == scenario]
        values.append(statistics.median(subset) if subset else float("nan"))
    plt.bar(scenarios, values)
    plt.ylabel("Median selected penalty factor")
    plt.title("Adaptive QUBO penalty by traffic event")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def phase3_demo(root, out, emergency_count=3, seed=0, reads=64, sweeps=900, target_feasibility=DEFAULT_TARGET_FEASIBILITY):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} already contains results. Choose a new --out directory.")
    out.mkdir(parents=True, exist_ok=True)
    base, ambulances, emergencies, hospitals, events = build_dynamic_scenario(root, emergency_count)
    rows = []
    penalty_traces = []
    scenario_payload = {
        "ambulances": ambulances,
        "emergencies": emergencies,
        "hospitals": hospitals,
        "events": events,
        "scope": "rolling-horizon traffic snapshots; no future-traffic prediction or continuous vehicle-motion model",
    }
    _write_json(scenario_payload, out / "phase3_scenario.json")

    for event in events:
        graph = snapshot(base, event["hour"], event["scenario"], event["event_edges"])
        options, _ = build_options(graph, ambulances, emergencies, hospitals)
        exact = _run_method(options, emergencies, "exact_dynamic", seed, reads, sweeps, target_feasibility)
        exact_score = exact["objective_score_s"]
        results = [exact]
        for method in ("fixed_hybrid", "adaptive_hybrid"):
            result = _run_method(options, emergencies, method, seed, reads, sweeps, target_feasibility)
            results.append(result)
            if method == "adaptive_hybrid":
                penalty_traces.append({
                    "epoch": event["epoch"],
                    "scenario": event["scenario"],
                    "selected_penalty_factor": result["selected_penalty_factor"],
                    "target_feasibility": result["target_feasibility"],
                    "trace": result["penalty_trace"],
                })
        for result in results:
            rows.append(_flatten(result, emergency_count, event, seed if result["method"] != "exact_dynamic" else -1, exact_score))
        chosen = results[-1] if results[-1]["status"] == "ok" else exact
        plot_dispatch(graph, chosen, out / f"phase3_{event['epoch']}_{event['scenario']}.png")
        print(
            f"Phase III epoch {event['epoch']} {event['scenario']}: "
            f"exact={exact_score:.3f}, adaptive_penalty={results[-1].get('selected_penalty_factor')}",
            flush=True,
        )

    _write_csv(rows, out / "phase3_event_results.csv")
    _write_json(penalty_traces, out / "adaptive_penalty_trace.json")
    _plot_runtime(rows, out / "phase3_runtime.png")
    _plot_penalties(rows, out / "phase3_penalty_selection.png")
    _write_json({
        "emergency_count": emergency_count,
        "seed": seed,
        "reads": reads,
        "sweeps": sweeps,
        "target_feasibility": target_feasibility,
        "penalty_factors": list(DEFAULT_PENALTY_FACTORS),
    }, out / "phase3_manifest.json")

    lines = [
        "# Phase III dynamic re-dispatch demo",
        "",
        "Phase III re-solves the Phase-II fleet assignment on successive traffic snapshots and compares a fixed QUBO penalty with an adaptive penalty chosen from raw-SA feasibility.",
        "",
        "| Epoch | Scenario | Method | Status | Objective (s) | Gap vs exact | Raw feasible fraction | Penalty factor | Deadline misses | Runtime (ms) |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gap = "N/A" if row["gap_pct"] is None else f"{row['gap_pct']:.3f}%"
        feasible = "N/A" if row["raw_feasible_fraction"] is None else f"{row['raw_feasible_fraction']:.3f}"
        penalty = "N/A" if row["selected_penalty_factor"] is None else f"{row['selected_penalty_factor']:.2f}"
        misses = "N/A" if row["deadline_misses"] is None else str(row["deadline_misses"])
        score = "N/A" if row["objective_score_s"] is None else f"{row['objective_score_s']:.3f}"
        lines.append(
            f"| {row['epoch']} | {row['scenario']} | {row['method']} | {row['status']} | {score} | {gap} | "
            f"{feasible} | {penalty} | {misses} | {row['runtime_ms']:.3f} |"
        )
    lines += [
        "",
        "Interpretation: adaptive tuning is evaluated by raw-SA feasibility, objective gap and runtime. The hybrid still retains a deterministic feasible candidate when annealing does not improve it.",
        "",
        "Limit: each epoch is a fresh rolling-horizon optimization using the current synthetic traffic snapshot. The demo does not predict future traffic or continuously simulate ambulance movement.",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def phase3_study(root, out, quick=False, seeds=(0, 1, 2), reads=64, sweeps=1000, target_feasibility=DEFAULT_TARGET_FEASIBILITY):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} already contains results. Choose a new --out directory.")
    out.mkdir(parents=True, exist_ok=True)
    sizes = (2, 3) if quick else (2, 3, 4)
    if quick:
        seeds = (0, 1)
        reads = 32
        sweeps = 500
    rows = []
    all_traces = []
    full = load_graph(Path(root) / "data" / "road_graph.json")
    base = _largest_scc_patch(full, 60)

    for size in sizes:
        ambulances, emergencies, hospitals = make_entities(base, size)
        primary_source = next(a["node"] for a in ambulances if a.get("available", True))
        primary_target = max(emergencies, key=lambda e: (int(e["severity"]), e["id"]))["node"]
        incident_edge = _event_edge(base, primary_source, primary_target)
        roadblock_edge = _safe_roadblock_edge(base, ambulances, emergencies, hospitals)
        events = [
            {"epoch": 0, "scenario": "normal", "hour": 12, "event_edges": []},
            {"epoch": 1, "scenario": "peak", "hour": 8, "event_edges": []},
            {"epoch": 2, "scenario": "accident", "hour": 8, "event_edges": [incident_edge] if incident_edge else []},
            {"epoch": 3, "scenario": "roadblock", "hour": 8, "event_edges": [roadblock_edge] if roadblock_edge else []},
        ]
        for event in events:
            graph = snapshot(base, event["hour"], event["scenario"], event["event_edges"])
            options, _ = build_options(graph, ambulances, emergencies, hospitals)
            exact = _run_method(options, emergencies, "exact_dynamic", -1, reads, sweeps, target_feasibility)
            exact_score = exact["objective_score_s"]
            rows.append(_flatten(exact, size, event, -1, exact_score))
            for seed in seeds:
                fixed = _run_method(options, emergencies, "fixed_hybrid", seed, reads, sweeps, target_feasibility)
                adaptive = _run_method(options, emergencies, "adaptive_hybrid", seed, reads, sweeps, target_feasibility)
                rows.append(_flatten(fixed, size, event, seed, exact_score))
                rows.append(_flatten(adaptive, size, event, seed, exact_score))
                all_traces.append({
                    "emergency_count": size,
                    "epoch": event["epoch"],
                    "scenario": event["scenario"],
                    "seed": seed,
                    "selected_penalty_factor": adaptive["selected_penalty_factor"],
                    "trace": adaptive["penalty_trace"],
                })
        print(f"Completed Phase III emergency count {size}", flush=True)

    _write_csv(rows, out / "phase3_raw_results.csv")
    _write_json(all_traces, out / "phase3_penalty_traces.json")
    _plot_runtime(rows, out / "phase3_runtime_scalability.png")
    _plot_penalties(rows, out / "phase3_penalty_selection.png")

    lines = [
        "# Phase III study summary",
        "",
        "Rolling-horizon fleet re-dispatch on public OSM geometry with synthetic traffic/events and adaptive QUBO penalty tuning.",
        "",
        "| Method | Runs | Solver failures | Median gap | Median runtime (ms) | Median raw feasible fraction | Median penalty factor |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in PHASE3_METHODS:
        subset = [row for row in rows if row["method"] == method]
        gaps = [row["gap_pct"] for row in subset if row["gap_pct"] is not None]
        feas = [row["raw_feasible_fraction"] for row in subset if row["raw_feasible_fraction"] is not None]
        penalties = [row["selected_penalty_factor"] for row in subset if row["selected_penalty_factor"] is not None]
        lines.append(
            f"| {method} | {len(subset)} | {sum(row['status']=='solver_failed' for row in subset)} | "
            f"{statistics.median(gaps):.3f}% | {statistics.median(row['runtime_ms'] for row in subset):.3f} | "
            f"{'N/A' if not feas else f'{statistics.median(feas):.3f}'} | "
            f"{'N/A' if not penalties else f'{statistics.median(penalties):.2f}'} |"
        )
    lines += [
        "",
        "Ablation: fixed_hybrid always uses penalty factor 2.0; adaptive_hybrid probes the declared candidate factors and selects the smallest factor reaching the target raw feasibility, or the best observed factor if none reaches it.",
        "",
        "Limitations: synthetic traffic/events and EMS entities; small-instance exact baseline; no real-time traffic feed, future-traffic prediction, continuous vehicle kinematics or clinical outcome model.",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    _write_json({
        "sizes": list(sizes),
        "event_sequence": ["normal", "peak", "accident", "roadblock"],
        "seeds": list(seeds),
        "reads": reads,
        "sweeps": sweeps,
        "target_feasibility": target_feasibility,
        "penalty_factors": list(DEFAULT_PENALTY_FACTORS),
        "scope": "Phase III rolling-horizon fleet re-dispatch + adaptive QUBO penalty tuning; OSM geometry with synthetic traffic/entities",
    }, out / "phase3_manifest.json")
    return rows
