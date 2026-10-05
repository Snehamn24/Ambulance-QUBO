"""Deterministic piecewise time-dependent snapshots, with optional disruptions."""
import hashlib
import math


def stable_fraction(key):
    return int(hashlib.sha256(str(key).encode()).hexdigest()[:8],16) / (2**32-1)


def snapshot(base, hour=8, scenario='normal', event_edges=(), traffic_seed=2026):
    if not math.isfinite(hour) or not 0 <= hour < 24:
        raise ValueError('hour must be between 0 inclusive and 24 exclusive')
    if scenario not in {'normal','peak','accident','roadblock'}:
        raise ValueError('Unknown traffic scenario')
    known = {k for _,_,k in base.edges(keys=True)}
    if set(event_edges)-known: raise ValueError('Unknown disruption edge')
    graph = base.copy()
    # Commute hours double the congestion contribution. Peak is an additional scenario stress.
    commute = 1.0 if 7 <= hour < 10 or 16 <= hour < 19 else .25
    for u,v,k,d in base.edges(keys=True,data=True):
        stress = 2 if scenario == 'peak' else 1
        multiplier = 1 + stress * commute * stable_fraction(f'{traffic_seed}:{k}')
        delay = 20 if base.nodes[v].get('signal',False) else 0
        if scenario == 'roadblock' and k in event_edges:
            graph.remove_edge(u,v,key=k); continue
        if scenario == 'accident' and k in event_edges: delay += 180
        graph[u][v][k]['time_s'] = d['base_time_s'] * multiplier + delay
    graph.graph.update(hour=hour,scenario=scenario)
    return graph
