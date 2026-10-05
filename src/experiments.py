"""Repeatable scenarios, fair solver comparisons, manifests and auditable raw results."""
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import sys
from importlib.metadata import version

import networkx as nx
from src.data import load_graph,study_patch,distance
from src.routes import baseline,edge_lookup,metrics
from src.solvers import METHODS,BASELINES,solve
from src.qubo import build_qubo
from src.traffic import snapshot
from src.plotting import comparisons,route_map

SCENARIOS=[('normal',12),('peak',8),('accident',8),('roadblock',8)]


def output_folder(path):
    path=Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'{path} already contains results. Choose a new --out directory to preserve them.')
    path.mkdir(parents=True,exist_ok=True)
    return path


def write_csv(rows,path):
    if not rows: raise ValueError('No rows to export')
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value) if isinstance(value,(list,dict)) else value for key,value in row.items()})


def manifest(root,config,path):
    files=[root/'main.py',root/'data/road_graph.json',root/'requirements.txt']+sorted((root/'src').glob('*.py'))
    payload={'created_at_utc':datetime.now(timezone.utc).isoformat(),'config':config,
             'platform':platform.platform(),'python':sys.version,
             'packages':{p:version(p) for p in ['networkx','numpy','scipy','matplotlib','dimod','dwave-samplers']},
             'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()},
             'runtime_definition':'solve() includes validation, reachability, preprocessing, QUBO construction, sampling and decoding; excludes shared map load, traffic construction, plotting and CSV I/O',
             'data_note':'Public OSM road geometry; class-based speeds and all traffic delays are simulated.'}
    Path(path).write_text(json.dumps(payload,indent=2),encoding='utf-8')


def select_pairs(graph,count,seed=17):
    candidates=[]
    for component in nx.strongly_connected_components(graph):
        nodes=sorted(component)
        for s in nodes:
            for t in nodes:
                if s != t and nx.shortest_path_length(graph,s,t) >= 2: candidates.append((s,t))
    random.Random(seed).shuffle(candidates)
    if len(candidates)<count: raise ValueError('Not enough reachable endpoint pairs in this patch')
    return candidates[:count]


def event_for_pair(graph,s,t):
    normal=snapshot(graph,12,'normal')
    route=baseline(normal,s,t,'dijkstra')
    return [route[len(route)//2]]


def annotate(result,graph,s,t,optimum,**labels):
    gap=None
    if result['status']=='ok' and optimum is not None and optimum>0:
        gap=100*(result['travel_time_s']-optimum)/optimum
        if gap < -1e-7: raise AssertionError('A method beat the same-cost exact optimum: investigate metrics')
        gap=max(0.,gap)
    return {**labels,'source':s,'target':t,'actual_nodes':len(graph),'edges':graph.number_of_edges(),
            **result,'optimum_time_s':optimum,'gap_pct':gap}


def study(root,out,sizes=(12,24,40),pairs=3,seeds=(0,1,2),reads=64,sweeps=1000,penalty_factor=2):
    out=output_folder(out); full=load_graph(root/'data/road_graph.json'); rows=[]; tasks=[]
    for size in sizes:
        base=study_patch(full,size)
        for pair_index,(s,t) in enumerate(select_pairs(base,pairs)):
            event=event_for_pair(base,s,t)
            for scenario,hour in SCENARIOS:
                graph=snapshot(base,hour,scenario,event)
                tasks.append({'requested_nodes':size,'pair':pair_index,'source':s,'target':t,'scenario':scenario,'hour':hour,'event_edges':event})
                exact={m:solve(graph,s,t,m) for m in BASELINES}
                optimum=exact['dijkstra']['travel_time_s']
                results=list(exact.values())
                for seed in seeds:
                    for method in ('sa','hybrid'):
                        results.append(solve(graph,s,t,method,seed,reads,sweeps,penalty_factor))
                for result in results:
                    rows.append(annotate(result,graph,s,t,optimum,requested_nodes=size,pair=pair_index,scenario=scenario,hour=hour))
                if pair_index==0 and scenario=='peak':
                    displayed=results[:3]+results[3:5]
                    route_map(graph,s,t,displayed,out/f'routes_{size}_peak.png',f'Public road patch: {len(graph)} nodes, peak scenario')
        print(f'Completed {len(base)}-node patch',flush=True)
    write_csv(rows,out/'raw_results.csv'); comparisons(rows,out)
    (out/'tasks.json').write_text(json.dumps(tasks,indent=2),encoding='utf-8')
    config={'sizes':list(sizes),'pairs_per_size':pairs,'seeds':list(seeds),'reads':reads,'sweeps':sweeps,
            'penalty_factor':penalty_factor,'candidate_k':4,'traffic_seed':2026,'scenarios':SCENARIOS}
    manifest(root,config,out/'manifest.json')
    summarize(rows,out/'SUMMARY.md')
    return rows


def summarize(rows,path):
    lines=['# Executed experiment summary','',
           'One OSM area near BMSCE; traffic and speed assumptions are simulated. This is an initial experiment, not proof of publication readiness.','',
           '| Method | Runs | No route | Solver failures | Median gap on successful reachable runs | Median runtime ms |',
           '|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        subset=[r for r in rows if r['method']==method]
        gaps=[r['gap_pct'] for r in subset if r['gap_pct'] is not None]
        gap=f'{statistics.median(gaps):.3f}%' if gaps else 'N/A'
        lines.append(f"| {method} | {len(subset)} | {sum(r['status']=='no_route' for r in subset)} | {sum(r['status']=='solver_failed' for r in subset)} | {gap} | {statistics.median(r['runtime_ms'] for r in subset):.3f} |")
    hybrid=[r for r in rows if r['method']=='hybrid' and r['status']=='ok']
    lines += ['',f"Hybrid retained a classical candidate in {sum(r['candidate_retained'] for r in hybrid)} of {len(hybrid)} successful runs. This is not evidence of annealing improvement.",
              '', 'Classical methods run once per task; stochastic methods run once per seed. Report failures alongside successful-route gaps. Equal reads/sweeps do not imply equal wall-clock budgets.',
              '', 'Consult raw_results.csv for raw feasible fractions, preprocessing and sampling timings, candidate costs and whether annealing improved a candidate.']
    Path(path).write_text('\n'.join(lines),encoding='utf-8')


def demo(root,out,public=False):
    from src.data import toy_graph
    out=output_folder(out)
    if public:
        base=study_patch(load_graph(root/'data/road_graph.json'),24); s,t=select_pairs(base,1)[0]
        graph=snapshot(base,8,'peak',event_for_pair(base,s,t))
    else: graph=toy_graph(); s,t='S','T'
    results=[solve(graph,s,t,m,seed=0,reads=64,sweeps=1000) for m in METHODS]
    route_map(graph,s,t,results,out/'routes.png','Public road comparison' if public else 'Toy QUBO correctness example')
    write_csv(results,out/'results.csv')
    bqm,info=build_qubo(graph,s,t)
    model={'source':s,'target':t,**info,'offset':bqm.offset,
           'linear':dict(bqm.linear),
           'quadratic':[{'u':u,'v':v,'bias':bias} for (u,v),bias in bqm.quadratic.items()]}
    (out/'qubo_model.json').write_text(json.dumps(model,indent=2),encoding='utf-8')
    manifest(root,{'mode':'demo','public':public,'source':s,'target':t,'reads':64,'sweeps':1000,'seed':0},out/'manifest.json')
    return results


def reroute(root,out):
    out=output_folder(out); base=study_patch(load_graph(root/'data/road_graph.json'),40)
    s,t=select_pairs(base,1)[0]; first=snapshot(base,8,'normal')
    initial=solve(first,s,t,'hybrid',seed=0)
    if initial['status']!='ok' or len(initial['path'])<2: raise ValueError('Need a route with two or more roads')
    traversed=initial['path'][0]; current=edge_lookup(first)[traversed][1]
    elapsed=metrics(first,[traversed])['travel_time_s']
    new_hour=(8+elapsed/3600)%24
    blocked=initial['path'][1]
    changed=snapshot(base,new_hour,'roadblock',[blocked])
    remaining=solve(changed,current,t,'hybrid',seed=0)
    payload={'original_source':s,'current_node':current,'target':t,'traversed_edges':[traversed],
             'elapsed_travel_time_s':elapsed,'new_snapshot_hour':new_hour,'blocked_edge':blocked,
             'initial':initial,'remaining':remaining,
             'total_estimated_travel_time_s':elapsed+remaining['travel_time_s'] if remaining['status']=='ok' else None,
             'note':'Reroute at next intersection, freezing weights for remaining route. No prediction of future traffic.'}
    (out/'reroute.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    route_map(changed,current,t,[remaining],out/'reroute.png','Reroute from the current intersection after a closure')
    manifest(root,{'mode':'reroute','source':s,'target':t,'blocked':blocked},out/'manifest.json')
    return payload


def penalty_study(root,out):
    out=output_folder(out); base=study_patch(load_graph(root/'data/road_graph.json'),24)
    s,t=select_pairs(base,1)[0]; graph=snapshot(base,8,'peak'); rows=[]
    optimum=metrics(graph,baseline(graph,s,t,'dijkstra'))['travel_time_s']
    for factor in (1.1,2.,5.):
        for seed in (0,1,2):
            result=solve(graph,s,t,'sa',seed,64,1000,factor)
            rows.append(annotate(result,graph,s,t,optimum,penalty_factor=factor))
    write_csv(rows,out/'penalty_results.csv')
    manifest(root,{'mode':'penalty_study','factors':[1.1,2,5],'seeds':[0,1,2],'reads':64,'sweeps':1000,'source':s,'target':t},out/'manifest.json')
    return rows
