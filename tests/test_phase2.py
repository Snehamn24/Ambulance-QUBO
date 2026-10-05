"""Phase II assignment-QUBO correctness tests."""
import itertools
import unittest
import dimod

from src.phase2 import (
    assignment_metrics,
    build_dispatch_qubo,
    exact_dispatch,
    greedy_candidate,
    solve_dispatch,
    validate_assignment,
)


def tiny_problem():
    emergencies = [
        {"id":"E1","severity":5,"response_deadline_s":90},
        {"id":"E2","severity":3,"response_deadline_s":180},
    ]
    rows = []
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
    for (a,e,s),cost in costs.items():
        h=s.split('#')[0]
        rows.append({
            "variable":f"x|{a}|{e}|{s}","ambulance_id":a,"emergency_id":e,"slot_id":s,
            "hospital_id":h,"score_s":float(cost),"normalized_cost":cost/scale,
            "response_time_s":float(cost),"response_deadline_s":1000.0,"transport_time_s":0.0,
        })
    return rows, emergencies


class Phase2Tests(unittest.TestCase):
    def test_exact_dispatch_is_expected_optimum(self):
        options,emergencies=tiny_problem()
        result=exact_dispatch(options,emergencies)
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['objective_score_s'],19.0)
        pairs={(o['ambulance_id'],o['emergency_id']) for o in result['assignment']}
        self.assertEqual(pairs,{('A1','E1'),('A2','E2')})

    def test_assignment_qubo_exact_solver_matches_classical(self):
        options,emergencies=tiny_problem(); classical=exact_dispatch(options,emergencies)
        bqm,_=build_dispatch_qubo(options,emergencies)
        best=dimod.ExactSolver().sample(bqm).first.sample
        decoded=validate_assignment([v for v,b in best.items() if b],options,emergencies)
        self.assertIsNotNone(decoded)
        self.assertAlmostEqual(assignment_metrics(decoded)['objective_score_s'],classical['objective_score_s'])

    def test_invalid_duplicate_ambulance_is_rejected(self):
        options,emergencies=tiny_problem()
        selected=['x|A1|E1|H1#1','x|A1|E2|H2#1']
        self.assertIsNone(validate_assignment(selected,options,emergencies))

    def test_hybrid_returns_feasible_assignment(self):
        options,emergencies=tiny_problem()
        candidate=greedy_candidate(options,emergencies)
        self.assertIsNotNone(candidate)
        result=solve_dispatch(options,emergencies,'hybrid_dispatch',seed=3,reads=64,sweeps=500)
        self.assertEqual(result['status'],'ok')
        self.assertIsNotNone(validate_assignment([o['variable'] for o in result['assignment']],options,emergencies))


if __name__=='__main__': unittest.main()
