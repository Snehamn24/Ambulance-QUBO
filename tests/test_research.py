"""Correctness tests: independent energy calculation, exact tiny optimum and edge cases."""
import itertools
import tempfile
from pathlib import Path
import unittest
import numpy as np
import dimod
import networkx as nx
from src.data import toy_graph,parse_osm,save_graph,load_graph,study_patch
from src.qubo import build_qubo
from src.routes import decode,metrics,baseline,edge_lookup
from src.solvers import solve,candidates,BASELINES
from src.traffic import snapshot
from src.quantum_demo import statevector

ROOT=Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def setUp(self): self.g=toy_graph()

    def test_qubo_energy_matches_independent_equation_for_all_bits(self):
        bqm,info=build_qubo(self.g,'S','T'); keys=list(bqm.variables)
        for bits in itertools.product((0,1),repeat=len(keys)):
            sample=dict(zip(keys,bits)); cost=0; flow=0; branch=0
            for _,_,k,d in self.g.edges(keys=True,data=True): cost+=d['time_s']/info['scale_seconds']*sample[k]
            for v in self.g:
                outgoing=sum(sample[k] for _,_,k in self.g.out_edges(v,keys=True))
                incoming=sum(sample[k] for _,_,k in self.g.in_edges(v,keys=True))
                b=1 if v=='S' else -1 if v=='T' else 0
                flow+=(outgoing-incoming-b)**2
                branch+=outgoing*(outgoing-1)/2+incoming*(incoming-1)/2
            self.assertAlmostEqual(bqm.energy(sample),cost+info['penalty']*(flow+branch),places=8)

    def test_exact_optimum_is_valid_and_equal_to_dijkstra(self):
        bqm,_=build_qubo(self.g,'S','T'); best=dimod.ExactSolver().sample(bqm).first
        route=decode(self.g,'S','T',[k for k,b in best.sample.items() if b])
        self.assertIsNotNone(route)
        self.assertEqual(metrics(self.g,route)['travel_time_s'],240)

    def test_classical_baselines_agree(self):
        for method in BASELINES: self.assertEqual(metrics(self.g,baseline(self.g,'S','T',method))['travel_time_s'],240)

    def test_sa_and_hybrid_return_only_valid_routes(self):
        for method in ('sa','hybrid'):
            result=solve(self.g,'S','T',method,seed=4,reads=128,sweeps=1000)
            self.assertEqual(result['status'],'ok')
            self.assertIsNotNone(decode(self.g,'S','T',result['path']))
            self.assertGreaterEqual(result['travel_time_s'],240)

    def test_cycles_and_extra_edges_are_rejected(self):
        self.g.add_edge('B','A',key='ba',edge_id='ba',length_m=100,base_time_s=10,time_s=10,geometry=[[0,.001],[.001,0]])
        self.assertIsNone(decode(self.g,'S','T',['sa','at','ab','ba']))
        self.assertIsNone(decode(self.g,'S','T',['sa','sb','at']))
        self.assertIsNone(decode(self.g,'S','T',['sa']))
        self.assertIsNone(decode(self.g,'S','T',['sa','sa','at']))

    def test_disconnected_cycle_is_rejected(self):
        for n in ('X','Y'): self.g.add_node(n,x=0,y=0,signal=False)
        self.g.add_edge('X','Y',key='xy',edge_id='xy'); self.g.add_edge('Y','X',key='yx',edge_id='yx')
        self.assertIsNone(decode(self.g,'S','T',['sa','at','xy','yx']))

    def test_parallel_roads_preserved(self):
        self.g.add_edge('S','A',key='fast',edge_id='fast',length_m=1000,base_time_s=1,time_s=1,geometry=[[0,0],[.001,0]])
        for method in BASELINES: self.assertIn('fast',baseline(self.g,'S','T',method))
        bqm,_=build_qubo(self.g,'S','T'); self.assertIn('fast',bqm.variables); self.assertIn('sa',bqm.variables)
        self.assertIsNone(decode(self.g,'S','T',['sa','fast','at']))

    def test_no_route_and_same_endpoint(self):
        for method in (*BASELINES,'sa','hybrid'):
            self.assertEqual(solve(self.g,'T','S',method)['status'],'no_route')
            self.assertEqual(solve(self.g,'S','S',method)['travel_time_s'],0)

    def test_traffic_is_reproducible_and_base_unchanged(self):
        a=snapshot(self.g,8,'peak'); b=snapshot(self.g,8,'peak')
        self.assertEqual(a['S']['A']['sa']['time_s'],b['S']['A']['sa']['time_s'])
        self.assertEqual(self.g['S']['A']['sa']['time_s'],120)
        self.assertGreater(a['S']['A']['sa']['time_s'],snapshot(self.g,12)['S']['A']['sa']['time_s'])

    def test_roadblock_and_accident(self):
        closed=snapshot(self.g,8,'roadblock',['at']); self.assertFalse(closed.has_edge('A','T','at'))
        self.assertNotIn('at',baseline(closed,'S','T','dijkstra'))
        ordinary=snapshot(self.g,8); accident=snapshot(self.g,8,'accident',['at'])
        self.assertAlmostEqual(accident['A']['T']['at']['time_s']-ordinary['A']['T']['at']['time_s'],180)
        with self.assertRaises(ValueError): snapshot(self.g,8,'roadblock',['unknown'])
        with self.assertRaises(ValueError): snapshot(self.g,24)

    def test_candidate_paths_and_subgraph_valid(self):
        reduced,seeds=candidates(self.g,'S','T')
        for seed in seeds: self.assertIsNotNone(decode(reduced,'S','T',seed))
        self.assertLessEqual(reduced.number_of_edges(),self.g.number_of_edges())

    def test_bad_weights(self):
        self.g['S']['A']['sa']['time_s']=float('nan')
        with self.assertRaises(ValueError): solve(self.g,'S','T','sa')

    def test_quantum_normalization_and_uniform_limit(self):
        energy=np.arange(32,dtype=float)/31
        for gamma,beta in ((0,0),(.8,.4),(3,1)):
            psi=statevector(energy,gamma,beta); self.assertAlmostEqual(float(np.sum(abs(psi)**2)),1,places=12)
        np.testing.assert_allclose(abs(statevector(energy,0,.7))**2,np.ones(32)/32,atol=1e-12)

    def test_quantum_single_qubit_matches_matrix(self):
        energy=np.array([0.,1.]); gamma,beta=.9,.3
        mixer=np.array([[np.cos(beta),-1j*np.sin(beta)],[-1j*np.sin(beta),np.cos(beta)]])
        expected=mixer @ (np.exp(-1j*gamma*energy)/np.sqrt(2))
        np.testing.assert_allclose(statevector(energy,gamma,beta),expected)

    def test_public_graph_roundtrip_and_patch(self):
        graph=load_graph(ROOT/'data/road_graph.json'); patch=study_patch(graph,24)
        self.assertEqual(len(patch),24)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'g.json'; save_graph(patch,path)
            again=load_graph(path); self.assertEqual(set(patch.edges(keys=True)),set(again.edges(keys=True)))
        prepared=snapshot(patch,8)
        comp=max(nx.strongly_connected_components(prepared),key=len); s,t=sorted(comp)[:2]
        times=[metrics(prepared,baseline(prepared,s,t,m))['travel_time_s'] for m in BASELINES]
        self.assertAlmostEqual(min(times),max(times))

    def test_parser_has_real_geometry_and_strict_positive_costs(self):
        graph=parse_osm(ROOT/'data/bmsce_area.osm')
        self.assertGreater(len(graph),40)
        self.assertTrue(nx.is_strongly_connected(graph))
        for _,_,_,d in graph.edges(keys=True,data=True):
            self.assertGreater(d['length_m'],0); self.assertGreater(d['base_time_s'],0)
            self.assertGreaterEqual(len(d['geometry']),2)


if __name__=='__main__': unittest.main()
