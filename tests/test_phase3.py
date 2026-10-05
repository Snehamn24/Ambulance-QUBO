"""Phase III adaptive-penalty and dynamic-event correctness tests."""
import unittest

from src.phase2 import validate_assignment
from src.phase3 import (
    DEFAULT_PENALTY_FACTORS,
    build_dynamic_scenario,
    choose_penalty_from_trace,
    solve_adaptive_hybrid,
)
from src.traffic import snapshot
from src.phase2 import build_options


def tiny_problem():
    emergencies = [
        {"id":"E1","severity":5,"response_deadline_s":90},
        {"id":"E2","severity":3,"response_deadline_s":180},
    ]
    costs = {
        ("A1","E1","H1#1"):10,
        ("A1","E1","H2#1"):12,
        ("A2","E1","H1#1"):30,
        ("A2","E1","H2#1"):32,
        ("A1","E2","H1#1"):20,
        ("A1","E2","H2#1"):22,
        ("A2","E2","H1#1"):8,
        ("A2","E2","H2#1"):9,
    }
    scale=max(costs.values())
    rows=[]
    for (ambulance, emergency, slot), cost in costs.items():
        rows.append({
            "variable":f"x|{ambulance}|{emergency}|{slot}",
            "ambulance_id":ambulance,
            "emergency_id":emergency,
            "slot_id":slot,
            "hospital_id":slot.split('#')[0],
            "score_s":float(cost),
            "normalized_cost":cost/scale,
            "response_time_s":float(cost),
            "response_deadline_s":1000.0,
            "transport_time_s":0.0,
        })
    return rows, emergencies


class Phase3Tests(unittest.TestCase):
    def test_penalty_selector_uses_smallest_factor_reaching_target(self):
        trace=[
            {"penalty_factor":1.25,"raw_feasible_fraction":0.20},
            {"penalty_factor":1.50,"raw_feasible_fraction":0.65},
            {"penalty_factor":2.00,"raw_feasible_fraction":0.90},
        ]
        self.assertEqual(choose_penalty_from_trace(trace,0.60),1.50)

    def test_penalty_selector_falls_back_to_best_observed(self):
        trace=[
            {"penalty_factor":1.25,"raw_feasible_fraction":0.20},
            {"penalty_factor":1.50,"raw_feasible_fraction":0.40},
            {"penalty_factor":2.00,"raw_feasible_fraction":0.40},
        ]
        self.assertEqual(choose_penalty_from_trace(trace,0.80),1.50)

    def test_adaptive_hybrid_returns_feasible_tiny_assignment(self):
        options, emergencies=tiny_problem()
        result=solve_adaptive_hybrid(
            options, emergencies, seed=7, reads=40, sweeps=300,
            probe_reads=12, probe_sweeps=120,
        )
        self.assertEqual(result["status"],"ok")
        self.assertIn(result["selected_penalty_factor"],DEFAULT_PENALTY_FACTORS)
        selected=[row["variable"] for row in result["assignment"]]
        self.assertIsNotNone(validate_assignment(selected,options,emergencies))

    def test_dynamic_scenario_keeps_each_epoch_dispatchable(self):
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        base, ambulances, emergencies, hospitals, events=build_dynamic_scenario(root,2)
        self.assertEqual([e["scenario"] for e in events],["normal","peak","accident","roadblock"])
        for event in events:
            graph=snapshot(base,event["hour"],event["scenario"],event["event_edges"])
            options,_=build_options(graph,ambulances,emergencies,hospitals)
            self.assertGreater(len(options),0)


if __name__=='__main__': unittest.main()
