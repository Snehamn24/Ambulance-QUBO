"""Routes are ordered edge IDs: parallel roads never disappear during decoding."""
import math
import networkx as nx
from src.data import distance


def edge_lookup(graph):
    return {k:(u,v,d) for u,v,k,d in graph.edges(keys=True,data=True)}


def check_inputs(graph,source,target):
    if source not in graph or target not in graph: raise ValueError('Unknown endpoint')
    for _,_,_,d in graph.edges(keys=True,data=True):
        if not math.isfinite(d.get('time_s',float('nan'))) or d['time_s'] <= 0:
            raise ValueError('Apply a traffic snapshot with positive finite time_s first')


def node_path_edges(graph,path,weight='time_s'):
    return [min(graph[u][v],key=lambda k:(graph[u][v][k][weight],str(k))) for u,v in zip(path,path[1:])]


def decode(graph,source,target,selected):
    """Return ordered edges only for one simple s-t route with no extra components."""
    lookup = edge_lookup(graph); selected = list(selected)
    if len(selected) != len(set(selected)) or set(selected)-set(lookup): return None
    if source == target: return [] if not selected else None
    outgoing, incoming = {},{}
    for k in selected:
        u,v,_ = lookup[k]
        if u in outgoing or v in incoming: return None
        outgoing[u] = (v,k); incoming[v] = u
    if source in incoming or target in outgoing: return None
    path=[]; seen={source}; current=source
    while current != target:
        if current not in outgoing: return None
        current,k = outgoing[current]; path.append(k)
        if current in seen: return None
        seen.add(current)
    return path if len(path) == len(selected) else None


def metrics(graph,edge_ids):
    lookup=edge_lookup(graph)
    return {'travel_time_s':sum(lookup[k][2]['time_s'] for k in edge_ids),
            'distance_m':sum(lookup[k][2]['length_m'] for k in edge_ids)}


def simple_graph(graph,weight):
    """Collapse parallel arcs only for candidate node paths, using the declared cost."""
    g=nx.DiGraph(); g.add_nodes_from(graph.nodes)
    for u,v,k,d in graph.edges(keys=True,data=True):
        if not g.has_edge(u,v) or d[weight] < g[u][v]['weight']:
            g.add_edge(u,v,weight=d[weight])
    return g


def baseline(graph,source,target,method):
    check_inputs(graph,source,target)
    if source == target: return []
    if method == 'dijkstra': path=nx.dijkstra_path(graph,source,target,weight='time_s')
    elif method == 'bellman_ford': path=nx.bellman_ford_path(graph,source,target,weight='time_s')
    elif method == 'astar':
        # alpha*d_geo is admissible: each edge cost >= alpha*its geodesic distance.
        ratios=[]
        for u,v,_,d in graph.edges(keys=True,data=True):
            geo=distance((graph.nodes[u]['x'],graph.nodes[u]['y']),(graph.nodes[v]['x'],graph.nodes[v]['y']))
            if geo > 0: ratios.append(d['time_s']/geo)
        alpha=min(ratios) if ratios else 0
        def heuristic(u,v):
            return alpha*distance((graph.nodes[u]['x'],graph.nodes[u]['y']),(graph.nodes[v]['x'],graph.nodes[v]['y']))
        path=nx.astar_path(graph,source,target,heuristic=heuristic,weight='time_s')
    else: raise ValueError('Unknown baseline')
    return node_path_edges(graph,path)
