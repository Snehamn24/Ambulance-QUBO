"""Phase II: fleet dispatch QUBO built on top of the Phase I route-cost engine.

The Phase I code optimizes one source-target route.  Phase II adds a small,
auditable dispatch layer: choose one available ambulance and one available
hospital capacity slot for every emergency.  Route travel times are computed on
the same traffic snapshot, then an assignment QUBO is solved by simulated
annealing and a feasible-candidate hybrid.

This is still a research simulation: hospital capacities, queue delays,
emergency severities and deadlines are synthetic inputs, not operational data.
"""
from __future__ import annotations

from itertools import combinations, permutations
from pathlib import Path
from time import perf_counter
import csv
import json
import math
import statistics

import dimod
import matplotlib.pyplot as plt
import networkx as nx
from dwave.samplers import SimulatedAnnealingSampler

from src.data import distance, load_graph, study_patch
from src.routes import baseline, edge_lookup, metrics
from src.traffic import snapshot


PHASE2_METHODS = ("exact_dispatch", "sa_dispatch", "hybrid_dispatch")
SEVERITY_WEIGHT = {1: 1.0, 2: 1.4, 3: 1.8, 4: 2.3, 5: 3.0}
DEADLINE_BY_SEVERITY = {1: 300.0, 2: 240.0, 3: 180.0, 4: 135.0, 5: 90.0}


def _farthest_nodes(graph, count):
    """Deterministic, spatially spread node selection for reproducible demos."""
    if len(graph) < count:
        raise ValueError(f"Need {count} distinct nodes but graph has only {len(graph)}")
    nodes = sorted(graph)
    selected = [nodes[0]]
    while len(selected) < count:
        def score(node):
            point = (graph.nodes[node]["x"], graph.nodes[node]["y"])
            return min(
                distance(point, (graph.nodes[s]["x"], graph.nodes[s]["y"]))
                for s in selected
            )
        remaining = [n for n in nodes if n not in selected]
        selected.append(max(remaining, key=lambda n: (score(n), str(n))))
    return selected


def _largest_scc_patch(full_graph, requested=60):
    patch = study_patch(full_graph, requested)
    component = max(nx.strongly_connected_components(patch), key=len)
    graph = patch.subgraph(component).copy()
    if len(graph) < 20:
        raise ValueError("Phase II needs a sufficiently connected public-road patch")
    return graph


def make_entities(graph, emergency_count=3):
    """Create deterministic synthetic fleet/emergency/hospital inputs on real OSM nodes.

    There are N+1 ambulances with one unavailable ambulance, N emergencies,
    and three hospitals with one unavailable hospital.  Available hospital
    slots equal N, so a feasible assignment exists.
    """
    if emergency_count < 2 or emergency_count > 5:
        raise ValueError("Use between 2 and 5 emergencies for this prototype")
    needed = (emergency_count + 1) + emergency_count + 3
    chosen = _farthest_nodes(graph, needed)
    cursor = 0
    ambulances = []
    for i in range(emergency_count + 1):
        ambulances.append({
            "id": f"A{i+1}",
            "node": chosen[cursor],
            "available": i < emergency_count,
        })
        cursor += 1
    severities = [5, 4, 3, 2, 1]
    emergencies = []
    for i in range(emergency_count):
        severity = severities[i]
        emergencies.append({
            "id": f"E{i+1}",
            "node": chosen[cursor],
            "severity": severity,
            "response_deadline_s": DEADLINE_BY_SEVERITY[severity],
        })
        cursor += 1
    cap1 = (emergency_count + 1) // 2
    cap2 = emergency_count - cap1
    hospitals = [
        {"id": "H1", "node": chosen[cursor], "available": True, "capacity": cap1, "queue_delay_s": 20.0},
        {"id": "H2", "node": chosen[cursor+1], "available": True, "capacity": cap2, "queue_delay_s": 40.0},
        {"id": "H3", "node": chosen[cursor+2], "available": False, "capacity": 1, "queue_delay_s": 10.0},
    ]
    return ambulances, emergencies, hospitals


def hospital_slots(hospitals):
    slots = []
    for hospital in hospitals:
        if not hospital.get("available", True):
            continue
        capacity = int(hospital.get("capacity", 0))
        if capacity < 0:
            raise ValueError("Hospital capacity cannot be negative")
        for index in range(capacity):
            slots.append({
                "slot_id": f"{hospital['id']}#{index+1}",
                "hospital_id": hospital["id"],
                "hospital_node": hospital["node"],
                "queue_delay_s": float(hospital.get("queue_delay_s", 0.0)),
            })
    return slots


def _route(graph, source, target):
    path = baseline(graph, source, target, "dijkstra")
    values = metrics(graph, path)
    return path, values["travel_time_s"], values["distance_m"]


def build_options(graph, ambulances, emergencies, hospitals):
    available_ambulances = [a for a in ambulances if a.get("available", True)]
    slots = hospital_slots(hospitals)
    if len(available_ambulances) < len(emergencies):
        raise ValueError("Not enough available ambulances for all emergencies")
    if len(slots) < len(emergencies):
        raise ValueError("Not enough available hospital capacity for all emergencies")

    # Cache repeated route legs.  Hospital route depends only on emergency and hospital node,
    # not on the capacity slot index.
    response_cache = {}
    transport_cache = {}
    options = []
    for ambulance in available_ambulances:
        for emergency in emergencies:
            rkey = (ambulance["node"], emergency["node"])
            if rkey not in response_cache:
                response_cache[rkey] = _route(graph, *rkey)
            response_path, response_time, response_distance = response_cache[rkey]
            severity = int(emergency["severity"])
            deadline = float(emergency["response_deadline_s"])
            tardiness = max(0.0, response_time - deadline)
            for slot in slots:
                tkey = (emergency["node"], slot["hospital_node"])
                if tkey not in transport_cache:
                    transport_cache[tkey] = _route(graph, *tkey)
                transport_path, transport_time, transport_distance = transport_cache[tkey]
                # Severity strongly weights ambulance arrival; patient transport and queue delay
                # still matter.  Deadline tardiness is an explicit soft penalty.
                score = (
                    SEVERITY_WEIGHT[severity] * response_time
                    + 0.70 * transport_time
                    + slot["queue_delay_s"]
                    + 2.50 * tardiness
                )
                variable = f"x|{ambulance['id']}|{emergency['id']}|{slot['slot_id']}"
                options.append({
                    "variable": variable,
                    "ambulance_id": ambulance["id"],
                    "ambulance_node": ambulance["node"],
                    "emergency_id": emergency["id"],
                    "emergency_node": emergency["node"],
                    "severity": severity,
                    "response_deadline_s": deadline,
                    "hospital_id": slot["hospital_id"],
                    "hospital_node": slot["hospital_node"],
                    "slot_id": slot["slot_id"],
                    "queue_delay_s": slot["queue_delay_s"],
                    "response_path": response_path,
                    "response_time_s": response_time,
                    "response_distance_m": response_distance,
                    "transport_path": transport_path,
                    "transport_time_s": transport_time,
                    "transport_distance_m": transport_distance,
                    "tardiness_s": tardiness,
                    "score_s": score,
                })
    if not options:
        raise ValueError("No dispatch options generated")
    scale = max(option["score_s"] for option in options)
    for option in options:
        option["normalized_cost"] = option["score_s"] / scale
    return options, slots


def build_dispatch_qubo(options, emergencies, penalty_factor=2.0):
    """Assignment QUBO.

    Objective: sum normalized dispatch costs.
    Hard constraints encoded as penalties:
      * each emergency receives exactly one assignment;
      * each ambulance is used at most once;
      * each hospital capacity slot is used at most once.
    """
    if not math.isfinite(penalty_factor) or penalty_factor <= 1:
        raise ValueError("penalty_factor must exceed 1")
    emergency_ids = [e["id"] for e in emergencies]
    if len(emergency_ids) != len(set(emergency_ids)):
        raise ValueError("Emergency IDs must be unique")
    option_map = {o["variable"]: o for o in options}
    if len(option_map) != len(options):
        raise ValueError("Dispatch variable names must be unique")

    # A feasible assignment has at most one normalized-cost unit per emergency.
    # P > number of emergencies makes a single hard-constraint violation more
    # expensive than the whole feasible objective range.
    penalty = penalty_factor * len(emergencies) + 1.0
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
    for option in options:
        bqm.add_variable(option["variable"], option["normalized_cost"])

    # Exactly one option per emergency: P * (sum x - 1)^2.
    for emergency_id in emergency_ids:
        variables = [o["variable"] for o in options if o["emergency_id"] == emergency_id]
        if not variables:
            raise ValueError(f"Emergency {emergency_id} has no feasible option")
        bqm.offset += penalty
        for variable in variables:
            bqm.add_linear(variable, -penalty)
        for first, second in combinations(variables, 2):
            bqm.add_quadratic(first, second, 2.0 * penalty)

    # At most one emergency per ambulance.
    for ambulance_id in sorted({o["ambulance_id"] for o in options}):
        variables = [o["variable"] for o in options if o["ambulance_id"] == ambulance_id]
        for first, second in combinations(variables, 2):
            bqm.add_quadratic(first, second, penalty)

    # At most one emergency per capacity slot.
    for slot_id in sorted({o["slot_id"] for o in options}):
        variables = [o["variable"] for o in options if o["slot_id"] == slot_id]
        for first, second in combinations(variables, 2):
            bqm.add_quadratic(first, second, penalty)

    return bqm, {
        "penalty": penalty,
        "qubo_variables": len(bqm.variables),
        "qubo_interactions": len(bqm.quadratic),
        "option_scale_s": max(o["score_s"] for o in options),
        "option_map": option_map,
    }


def validate_assignment(selected, options, emergencies):
    option_map = {o["variable"]: o for o in options}
    selected = list(selected)
    if len(selected) != len(set(selected)) or any(v not in option_map for v in selected):
        return None
    chosen = [option_map[v] for v in selected]
    expected = {e["id"] for e in emergencies}
    emergency_ids = [o["emergency_id"] for o in chosen]
    if set(emergency_ids) != expected or len(emergency_ids) != len(expected):
        return None
    ambulance_ids = [o["ambulance_id"] for o in chosen]
    slot_ids = [o["slot_id"] for o in chosen]
    if len(set(ambulance_ids)) != len(ambulance_ids):
        return None
    if len(set(slot_ids)) != len(slot_ids):
        return None
    return sorted(chosen, key=lambda o: o["emergency_id"])


def assignment_metrics(chosen):
    if not chosen:
        return {
            "objective_score_s": 0.0,
            "total_response_time_s": 0.0,
            "mean_response_time_s": 0.0,
            "max_response_time_s": 0.0,
            "total_transport_time_s": 0.0,
            "deadline_misses": 0,
        }
    responses = [o["response_time_s"] for o in chosen]
    return {
        "objective_score_s": sum(o["score_s"] for o in chosen),
        "total_response_time_s": sum(responses),
        "mean_response_time_s": statistics.mean(responses),
        "max_response_time_s": max(responses),
        "total_transport_time_s": sum(o["transport_time_s"] for o in chosen),
        "deadline_misses": sum(o["response_time_s"] > o["response_deadline_s"] for o in chosen),
    }


def exact_dispatch(options, emergencies):
    """Small exact baseline by enumerating ambulance/slot permutations."""
    start = perf_counter()
    emergency_ids = [e["id"] for e in emergencies]
    ambulance_ids = sorted({o["ambulance_id"] for o in options})
    slot_ids = sorted({o["slot_id"] for o in options})
    if len(ambulance_ids) < len(emergency_ids) or len(slot_ids) < len(emergency_ids):
        raise ValueError("Insufficient dispatch resources")
    lookup = {(o["ambulance_id"], o["emergency_id"], o["slot_id"]): o for o in options}
    best = None
    best_cost = float("inf")
    for ambulance_perm in permutations(ambulance_ids, len(emergency_ids)):
        for slot_perm in permutations(slot_ids, len(emergency_ids)):
            chosen = []
            feasible = True
            for emergency_id, ambulance_id, slot_id in zip(emergency_ids, ambulance_perm, slot_perm):
                option = lookup.get((ambulance_id, emergency_id, slot_id))
                if option is None:
                    feasible = False
                    break
                chosen.append(option)
            if feasible:
                cost = sum(o["score_s"] for o in chosen)
                if cost < best_cost - 1e-9:
                    best, best_cost = chosen, cost
    if best is None:
        raise ValueError("No feasible exact dispatch assignment")
    result = {
        "method": "exact_dispatch",
        "status": "ok",
        "assignment": sorted(best, key=lambda o: o["emergency_id"]),
        "runtime_ms": (perf_counter() - start) * 1000,
        "raw_feasible_fraction": None,
        "candidate_retained": False,
        "improved_over_candidate": False,
        **assignment_metrics(best),
    }
    return result


def greedy_candidate(options, emergencies):
    """Severity-first feasible initialization for the hybrid solver."""
    chosen = []
    used_ambulances = set()
    used_slots = set()
    emergency_order = sorted(emergencies, key=lambda e: (-int(e["severity"]), float(e["response_deadline_s"]), e["id"]))
    for emergency in emergency_order:
        available = [
            o for o in options
            if o["emergency_id"] == emergency["id"]
            and o["ambulance_id"] not in used_ambulances
            and o["slot_id"] not in used_slots
        ]
        if not available:
            return None
        best = min(available, key=lambda o: (o["score_s"], o["ambulance_id"], o["slot_id"]))
        chosen.append(best)
        used_ambulances.add(best["ambulance_id"])
        used_slots.add(best["slot_id"])
    return sorted(chosen, key=lambda o: o["emergency_id"])


def solve_dispatch(options, emergencies, method="sa_dispatch", seed=0, reads=64, sweeps=1000, penalty_factor=2.0):
    if method not in {"sa_dispatch", "hybrid_dispatch"}:
        raise ValueError("Unknown Phase II dispatch solver")
    if reads < 1 or sweeps < 1:
        raise ValueError("reads and sweeps must be positive")
    start = perf_counter()
    bqm, info = build_dispatch_qubo(options, emergencies, penalty_factor)
    candidate = greedy_candidate(options, emergencies) if method == "hybrid_dispatch" else None
    kwargs = {}
    if candidate is not None:
        selected = {o["variable"] for o in candidate}
        state = {variable: int(variable in selected) for variable in bqm.variables}
        kwargs = {"initial_states": [state], "initial_states_generator": "random"}
    samples = SimulatedAnnealingSampler().sample(
        bqm,
        num_reads=reads,
        num_sweeps=sweeps,
        seed=seed,
        **kwargs,
    )
    feasible = []
    feasible_reads = 0
    total_reads = 0
    for record in samples.data(["sample", "num_occurrences"]):
        occurrences = int(record.num_occurrences)
        total_reads += occurrences
        decoded = validate_assignment(
            [variable for variable, bit in record.sample.items() if bit],
            options,
            emergencies,
        )
        if decoded is not None:
            feasible.append(decoded)
            feasible_reads += occurrences
    best = min(feasible, key=lambda assignment: assignment_metrics(assignment)["objective_score_s"]) if feasible else None
    raw_best_score = assignment_metrics(best)["objective_score_s"] if best is not None else None
    candidate_retained = False
    improved_over_candidate = False
    candidate_score = assignment_metrics(candidate)["objective_score_s"] if candidate is not None else None
    if candidate is not None:
        if best is None or raw_best_score >= candidate_score - 1e-9:
            best = candidate
            candidate_retained = True
        else:
            improved_over_candidate = True
    status = "ok" if best is not None else "solver_failed"
    result = {
        "method": method,
        "status": status,
        "assignment": best or [],
        "runtime_ms": (perf_counter() - start) * 1000,
        "sampling_reads": total_reads,
        "raw_feasible_fraction": feasible_reads / total_reads if total_reads else 0.0,
        "candidate_score_s": candidate_score,
        "candidate_retained": candidate_retained,
        "improved_over_candidate": improved_over_candidate,
        "qubo_variables": info["qubo_variables"],
        "qubo_interactions": info["qubo_interactions"],
        "penalty": info["penalty"],
        "seed": seed,
    }
    if best is not None:
        result.update(assignment_metrics(best))
    else:
        result.update({
            "objective_score_s": None,
            "total_response_time_s": None,
            "mean_response_time_s": None,
            "max_response_time_s": None,
            "total_transport_time_s": None,
            "deadline_misses": None,
        })
    return result


def _serializable_result(result):
    copy = dict(result)
    copy["assignment"] = [
        {k: v for k, v in option.items() if k != "normalized_cost"}
        for option in result.get("assignment", [])
    ]
    return copy


def _event_edge(base_graph, ambulance_node, emergency_node):
    ordinary = snapshot(base_graph, 12, "normal")
    route = baseline(ordinary, ambulance_node, emergency_node, "dijkstra")
    return [route[len(route)//2]] if route else []


def build_public_scenario(root, emergency_count=3, traffic_scenario="peak", hour=8):
    full = load_graph(Path(root) / "data" / "road_graph.json")
    base = _largest_scc_patch(full, 60)
    ambulances, emergencies, hospitals = make_entities(base, emergency_count)
    event = []
    if traffic_scenario in {"accident", "roadblock"}:
        event = _event_edge(base, ambulances[0]["node"], emergencies[0]["node"])
    graph = snapshot(base, hour, traffic_scenario, event)
    return graph, ambulances, emergencies, hospitals, event


def plot_dispatch(graph, result, path):
    path = Path(path)
    plt.figure(figsize=(10, 8))
    # Road background.
    for _, _, _, data in graph.edges(keys=True, data=True):
        geometry = data.get("geometry") or []
        if len(geometry) >= 2:
            xs = [p[0] for p in geometry]
            ys = [p[1] for p in geometry]
            plt.plot(xs, ys, linewidth=0.5, alpha=0.18)
    lookup = edge_lookup(graph)
    for index, option in enumerate(result.get("assignment", []), start=1):
        for edge_id in option["response_path"]:
            geometry = lookup[edge_id][2].get("geometry", [])
            if geometry:
                plt.plot([p[0] for p in geometry], [p[1] for p in geometry], linewidth=2.4, label=f"E{index} response" if edge_id == option["response_path"][0] else None)
        for edge_id in option["transport_path"]:
            geometry = lookup[edge_id][2].get("geometry", [])
            if geometry:
                plt.plot([p[0] for p in geometry], [p[1] for p in geometry], linewidth=1.6, linestyle="--", label=f"E{index} hospital" if edge_id == option["transport_path"][0] else None)
        for node, marker, label in [
            (option["ambulance_node"], "s", option["ambulance_id"]),
            (option["emergency_node"], "X", option["emergency_id"]),
            (option["hospital_node"], "^", option["hospital_id"]),
        ]:
            plt.scatter(graph.nodes[node]["x"], graph.nodes[node]["y"], marker=marker, s=70)
            plt.text(graph.nodes[node]["x"], graph.nodes[node]["y"], label, fontsize=8)
    plt.title(f"Phase II fleet dispatch: {result['method']}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _write_json(payload, path):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def phase2_demo(root, out, emergency_count=3, seed=0, reads=96, sweeps=1200):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} already contains results. Choose a new --out directory.")
    out.mkdir(parents=True, exist_ok=True)
    graph, ambulances, emergencies, hospitals, event = build_public_scenario(root, emergency_count, "peak", 8)
    options, slots = build_options(graph, ambulances, emergencies, hospitals)
    exact = exact_dispatch(options, emergencies)
    sa = solve_dispatch(options, emergencies, "sa_dispatch", seed, reads, sweeps)
    hybrid = solve_dispatch(options, emergencies, "hybrid_dispatch", seed, reads, sweeps)
    results = [exact, sa, hybrid]
    optimum = exact["objective_score_s"]
    for result in results:
        result["gap_pct"] = None if result["status"] != "ok" else max(0.0, 100.0 * (result["objective_score_s"] - optimum) / optimum)
    bqm, info = build_dispatch_qubo(options, emergencies)
    _write_json({
        "traffic": {"scenario": "peak", "hour": 8, "event_edges": event},
        "ambulances": ambulances,
        "emergencies": emergencies,
        "hospitals": hospitals,
        "hospital_slots": slots,
    }, out / "scenario.json")
    _write_json([_serializable_result(r) for r in results], out / "dispatch_results.json")
    _write_json({
        "penalty": info["penalty"],
        "qubo_variables": info["qubo_variables"],
        "qubo_interactions": info["qubo_interactions"],
        "offset": bqm.offset,
        "linear": dict(bqm.linear),
        "quadratic": [{"u": u, "v": v, "bias": bias} for (u, v), bias in bqm.quadratic.items()],
    }, out / "dispatch_qubo.json")
    chosen_for_plot = hybrid if hybrid["status"] == "ok" else exact
    plot_dispatch(graph, chosen_for_plot, out / "phase2_dispatch.png")
    summary = [
        "# Phase II dispatch demo",
        "",
        "Real OSM road geometry with synthetic traffic, emergency severity/deadlines, ambulance availability and hospital capacity/availability.",
        "",
        "| Method | Status | Objective score (s) | Gap vs exact | Mean response (s) | Deadline misses | Runtime (ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        gap = "N/A" if r["gap_pct"] is None else f"{r['gap_pct']:.3f}%"
        score = "N/A" if r["objective_score_s"] is None else f"{r['objective_score_s']:.3f}"
        mean = "N/A" if r["mean_response_time_s"] is None else f"{r['mean_response_time_s']:.3f}"
        misses = "N/A" if r["deadline_misses"] is None else str(r["deadline_misses"])
        summary.append(f"| {r['method']} | {r['status']} | {score} | {gap} | {mean} | {misses} | {r['runtime_ms']:.3f} |")
    summary += [
        "",
        "The exact dispatch is the small-instance reference optimum. Raw SA may fail or return a higher-cost feasible assignment. The hybrid retains a severity-first feasible candidate when annealing does not improve it.",
        "",
        "No quantum advantage is claimed. Phase II is an assignment/fleet extension over the Phase I route-cost engine.",
    ]
    (out / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    return results


def _flatten_study_result(result, fleet_size, scenario, seed, optimum):
    gap = None
    if result["status"] == "ok":
        gap = max(0.0, 100.0 * (result["objective_score_s"] - optimum) / optimum)
    return {
        "fleet_size": fleet_size,
        "scenario": scenario,
        "seed": seed,
        "method": result["method"],
        "status": result["status"],
        "objective_score_s": result["objective_score_s"],
        "gap_pct": gap,
        "mean_response_time_s": result["mean_response_time_s"],
        "max_response_time_s": result["max_response_time_s"],
        "deadline_misses": result["deadline_misses"],
        "runtime_ms": result["runtime_ms"],
        "raw_feasible_fraction": result.get("raw_feasible_fraction"),
        "candidate_retained": result.get("candidate_retained", False),
        "improved_over_candidate": result.get("improved_over_candidate", False),
        "qubo_variables": result.get("qubo_variables", 0),
        "qubo_interactions": result.get("qubo_interactions", 0),
    }


def _write_csv(rows, path):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_phase2_scalability(rows, path):
    sizes = sorted({r["fleet_size"] for r in rows})
    plt.figure(figsize=(8, 5))
    for method in PHASE2_METHODS:
        values = []
        for size in sizes:
            subset = [r["runtime_ms"] for r in rows if r["fleet_size"] == size and r["method"] == method]
            values.append(statistics.median(subset) if subset else float("nan"))
        plt.plot(sizes, values, marker="o", label=method)
    plt.xlabel("Number of simultaneous emergencies")
    plt.ylabel("Median runtime (ms)")
    plt.title("Phase II dispatch runtime scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def phase2_study(root, out, quick=False, seeds=(0, 1, 2), reads=64, sweeps=1000):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} already contains results. Choose a new --out directory.")
    out.mkdir(parents=True, exist_ok=True)
    sizes = (2, 3) if quick else (2, 3, 4)
    scenarios = (("normal", 12), ("peak", 8), ("accident", 8))
    if quick:
        seeds, reads, sweeps = (0, 1), 32, 500
    rows = []
    full = load_graph(Path(root) / "data" / "road_graph.json")
    base = _largest_scc_patch(full, 60)
    for size in sizes:
        ambulances, emergencies, hospitals = make_entities(base, size)
        accident_edge = _event_edge(base, ambulances[0]["node"], emergencies[0]["node"])
        for scenario, hour in scenarios:
            event = accident_edge if scenario == "accident" else []
            graph = snapshot(base, hour, scenario, event)
            options, _ = build_options(graph, ambulances, emergencies, hospitals)
            exact = exact_dispatch(options, emergencies)
            optimum = exact["objective_score_s"]
            rows.append(_flatten_study_result(exact, size, scenario, -1, optimum))
            for seed in seeds:
                for method in ("sa_dispatch", "hybrid_dispatch"):
                    result = solve_dispatch(options, emergencies, method, seed, reads, sweeps)
                    rows.append(_flatten_study_result(result, size, scenario, seed, optimum))
        print(f"Completed Phase II fleet size {size}", flush=True)
    _write_csv(rows, out / "phase2_raw_results.csv")
    _plot_phase2_scalability(rows, out / "phase2_runtime_scalability.png")
    lines = [
        "# Phase II study summary",
        "",
        "Synthetic fleet/emergency/hospital inputs on the same public OSM road area. Traffic remains simulated.",
        "",
        "| Method | Runs | Solver failures | Median gap on successful runs | Median runtime (ms) | Median deadline misses |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in PHASE2_METHODS:
        subset = [r for r in rows if r["method"] == method]
        gaps = [r["gap_pct"] for r in subset if r["gap_pct"] is not None]
        misses = [r["deadline_misses"] for r in subset if r["deadline_misses"] is not None]
        lines.append(
            f"| {method} | {len(subset)} | {sum(r['status']=='solver_failed' for r in subset)} | "
            f"{statistics.median(gaps):.3f}% | {statistics.median(r['runtime_ms'] for r in subset):.3f} | "
            f"{statistics.median(misses):.1f} |"
        )
    hybrid = [r for r in rows if r["method"] == "hybrid_dispatch" and r["status"] == "ok"]
    lines += [
        "",
        f"Hybrid retained its feasible candidate in {sum(r['candidate_retained'] for r in hybrid)} of {len(hybrid)} successful runs.",
        "",
        "Interpret assignment quality together with feasibility, response-time metrics and runtime. This is a small research benchmark, not an operational EMS evaluation.",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    _write_json({
        "sizes": list(sizes),
        "scenarios": scenarios,
        "seeds": list(seeds),
        "reads": reads,
        "sweeps": sweeps,
        "scope": "Phase II fleet dispatch; OSM geometry, synthetic traffic/entities",
    }, out / "phase2_manifest.json")
    return rows
