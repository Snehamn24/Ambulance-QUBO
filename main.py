"""Command-line entry point. Run python main.py --help for Phase I, II and III steps."""
import argparse
from pathlib import Path
import json
from src.data import download,parse_osm,save_graph
from src.experiments import demo,study,reroute,penalty_study,output_folder,manifest
from src.quantum_demo import run_demo
from src.phase2 import phase2_demo,phase2_study
from src.phase3 import phase3_demo,phase3_study

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','demo','experiment','reroute','quantum-demo','penalty-study','phase2-demo','phase2-study','phase3-demo','phase3-study'])
    parser.add_argument('--out',type=Path,default=None,help='New output folder; existing nonempty folders are preserved')
    parser.add_argument('--public',action='store_true',help='Use public road map for demo instead of toy fixture')
    parser.add_argument('--quick',action='store_true',help='Run the smaller validation/study configuration for the selected phase')
    parser.add_argument('--sizes',type=int,nargs='+',default=[12,24,40])
    parser.add_argument('--pairs',type=int,default=3)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--reads',type=int,default=64)
    parser.add_argument('--sweeps',type=int,default=1000)
    parser.add_argument('--emergencies',type=int,default=3,help='Phase II/III demo simultaneous emergency count (2-5)')
    parser.add_argument('--target-feasibility',type=float,default=0.60,help='Phase III adaptive-penalty raw-SA feasibility target in [0,1]')
    args=parser.parse_args(); out=args.out or ROOT/'outputs'/args.command
    try:
        if args.command=='prepare':
            raw=ROOT/'data/bmsce_area.osm'; download(raw)
            graph=parse_osm(raw); save_graph(graph,ROOT/'data/road_graph.json',raw)
            print(f'Public road graph saved: {len(graph)} nodes, {graph.number_of_edges()} directed roads')
        elif args.command=='demo':
            for result in demo(ROOT,out,args.public):
                print(result['method'],result['status'],result['travel_time_s'])
        elif args.command=='experiment':
            if args.pairs<1 or not args.seeds or any(n<4 for n in args.sizes): raise ValueError('Invalid experiment dimensions')
            if any(seed<0 or seed>=2**31 for seed in args.seeds): raise ValueError('Use integer seeds from 0 to 2^31-1')
            if args.quick: study(ROOT,out,sizes=(12,24),pairs=2,seeds=(0,1),reads=24,sweeps=500)
            else: study(ROOT,out,sizes=args.sizes,pairs=args.pairs,seeds=args.seeds,reads=args.reads,sweeps=args.sweeps)
        elif args.command=='reroute': print(json.dumps(reroute(ROOT,out),indent=2))
        elif args.command=='quantum-demo':
            output_folder(out); print(json.dumps(run_demo(out/'quantum_demo.json'),indent=2))
            manifest(ROOT,{'mode':'quantum-demo','seed':2026,'qubits':5,'depth':1},out/'manifest.json')
        elif args.command=='penalty-study': penalty_study(ROOT,out)
        elif args.command=='phase2-demo':
            results=phase2_demo(ROOT,out,args.emergencies,seed=args.seeds[0] if args.seeds else 0,reads=args.reads,sweeps=args.sweeps)
            for result in results:
                print(result['method'],result['status'],result.get('objective_score_s'),result.get('deadline_misses'))
        elif args.command=='phase2-study':
            phase2_study(ROOT,out,args.quick,args.seeds,args.reads,args.sweeps)
        elif args.command=='phase3-demo':
            if not 0 <= args.target_feasibility <= 1: raise ValueError('--target-feasibility must be between 0 and 1')
            rows=phase3_demo(ROOT,out,args.emergencies,seed=args.seeds[0] if args.seeds else 0,reads=args.reads,sweeps=args.sweeps,target_feasibility=args.target_feasibility)
            for row in rows:
                print(row['epoch'],row['scenario'],row['method'],row['status'],row.get('objective_score_s'),row.get('selected_penalty_factor'))
        else:
            if not 0 <= args.target_feasibility <= 1: raise ValueError('--target-feasibility must be between 0 and 1')
            phase3_study(ROOT,out,args.quick,args.seeds,args.reads,args.sweeps,args.target_feasibility)
        if args.command!='prepare': print('Results:',out.resolve())
    except (ValueError,OSError) as error:
        parser.exit(2,f'Error: {error}\n')


if __name__=='__main__': main()
