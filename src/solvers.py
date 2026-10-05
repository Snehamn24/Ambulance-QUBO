"""Classical baselines, raw QUBO annealing and a candidate-subgraph hybrid."""
from itertools import islice
from time import perf_counter
import networkx as nx
from dwave.samplers import SimulatedAnnealingSampler
from src.qubo import build_qubo
from src.routes import baseline, check_inputs, node_path_edges, simple_graph, metrics, decode

BASELINES=('dijkstra','astar','bellman_ford')
METHODS=BASELINES+('sa','hybrid')


def candidates(graph,source,target,k=4):
    """Distance-based k paths + BFS path. No time-optimal route is supplied as a seed."""
    simple=simple_graph(graph,'length_m')
    paths=list(islice(nx.shortest_simple_paths(simple,source,target,weight='weight'),k))
    paths.append(nx.shortest_path(simple,source,target))
    pairs={pair for path in paths for pair in zip(path,path[1:])}
    chosen=[(u,v,key) for u,v,key in graph.edges(keys=True) if (u,v) in pairs]
    reduced=graph.edge_subgraph(chosen).copy()
    # Pick the least-time parallel arc along each already chosen node sequence.
    seeds=[node_path_edges(graph,p) for p in paths]
    return reduced,seeds


def solve(graph,source,target,method,seed=0,reads=32,sweeps=1000,penalty_factor=2,k=4):
    if method not in METHODS: raise ValueError('Unknown solver')
    if reads < 1 or sweeps < 1 or k < 1: raise ValueError('Solver budgets must be positive')
    start=perf_counter(); check_inputs(graph,source,target)
    out={'method':method,'status':'ok','path':[], 'travel_time_s':None,'distance_m':None,
         'runtime_ms':None,'preprocess_ms':0.,'qubo_ms':0.,'sampling_ms':0.,'validation_ms':0.,
         'qubo_variables':0,'qubo_interactions':0,'solver_edges':graph.number_of_edges(),
         'raw_feasible_fraction':None,'candidate_time_s':None,'raw_sa_time_s':None,
         'candidate_retained':False,'improved_over_candidate':False,'seed':seed}
    if source == target:
        out.update(metrics(graph,[])); out['runtime_ms']=(perf_counter()-start)*1000; return out
    if not nx.has_path(graph,source,target):
        out.update(status='no_route',runtime_ms=(perf_counter()-start)*1000); return out
    if method in BASELINES:
        route=baseline(graph,source,target,method)
        out.update(path=route,**metrics(graph,route)); out['runtime_ms']=(perf_counter()-start)*1000
        return out
    pre=perf_counter(); working=graph; seed_routes=[]
    if method == 'hybrid': working,seed_routes=candidates(graph,source,target,k)
    out['preprocess_ms']=(perf_counter()-pre)*1000
    begin=perf_counter(); bqm,info=build_qubo(working,source,target,penalty_factor)
    out['qubo_ms']=(perf_counter()-begin)*1000
    out.update({key:info[key] for key in ('qubo_variables','qubo_interactions')})
    out['solver_edges']=working.number_of_edges()
    kwargs={}
    if seed_routes:
        # Include the best candidate even when num_reads is smaller than candidate count.
        seed_routes.sort(key=lambda p:metrics(graph,p)['travel_time_s'])
        states=[{v:int(v in set(p)) for v in bqm.variables} for p in seed_routes[:reads]]
        kwargs={'initial_states':states,'initial_states_generator':'random'}
    begin=perf_counter()
    samples=SimulatedAnnealingSampler().sample(bqm,num_reads=reads,num_sweeps=sweeps,seed=seed,**kwargs)
    out['sampling_ms']=(perf_counter()-begin)*1000
    begin=perf_counter(); feasible=[]; valid_reads=0
    for record in samples.data(['sample','num_occurrences']):
        route=decode(working,source,target,[v for v,b in record.sample.items() if b])
        if route is not None:
            feasible.append(route); valid_reads += int(record.num_occurrences)
    out['raw_feasible_fraction']=valid_reads/sum(int(n) for n in samples.record.num_occurrences)
    best=min(feasible,key=lambda p:metrics(graph,p)['travel_time_s']) if feasible else None
    if best is not None: out['raw_sa_time_s']=metrics(graph,best)['travel_time_s']
    if seed_routes:
        candidate=seed_routes[0]; candidate_cost=metrics(graph,candidate)['travel_time_s']
        out['candidate_time_s']=candidate_cost
        if best is None or out['raw_sa_time_s'] >= candidate_cost-1e-9:
            best=candidate; out['candidate_retained']=True
        else: out['improved_over_candidate']=True
    if best is None: out['status']='solver_failed'
    else: out.update(path=best,**metrics(graph,best))
    out['validation_ms']=(perf_counter()-begin)*1000
    out['runtime_ms']=(perf_counter()-start)*1000
    return out
