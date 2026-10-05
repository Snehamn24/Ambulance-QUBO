"""Edge-based QUBO: travel cost + flow balance + no-branch penalties."""
from itertools import combinations
import math
import dimod
import networkx as nx
from src.routes import node_path_edges, metrics, check_inputs


def build_qubo(graph,source,target,penalty_factor=2.0):
    check_inputs(graph,source,target)
    if source == target: raise ValueError('Handle identical endpoints before constructing QUBO')
    if not math.isfinite(penalty_factor) or penalty_factor <= 1:
        raise ValueError('penalty_factor must exceed 1')
    # A BFS feasible route provides a penalty bound without solving the time objective first.
    reference=node_path_edges(graph,nx.shortest_path(graph,source,target))
    scale=max(d['time_s'] for _,_,_,d in graph.edges(keys=True,data=True))
    bound=metrics(graph,reference)['travel_time_s']/scale
    penalty=penalty_factor*bound+1
    bqm=dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
    for _,_,k,d in graph.edges(keys=True,data=True):
        bqm.add_variable(k,d['time_s']/scale)
    for node in graph:
        # A*(sum a_e*x_e - b)^2, using x_e^2=x_e for binary variables.
        terms=[(k,1) for _,_,k in graph.out_edges(node,keys=True)]
        terms += [(k,-1) for _,_,k in graph.in_edges(node,keys=True)]
        b=1 if node == source else -1 if node == target else 0
        bqm.offset += penalty*b*b
        for k,a in terms: bqm.add_linear(k,penalty*(a*a-2*b*a))
        for (k,a),(j,c) in combinations(terms,2): bqm.add_quadratic(k,j,2*penalty*a*c)
        # Pairwise penalties prevent choosing multiple outgoing or incoming edges.
        for arcs in (list(graph.out_edges(node,keys=True)), list(graph.in_edges(node,keys=True))):
            for first,second in combinations(arcs,2):
                bqm.add_quadratic(first[2],second[2],penalty)
    return bqm, {'penalty':penalty,'scale_seconds':scale,'reference':reference,
                 'qubo_variables':len(bqm.variables),'qubo_interactions':len(bqm.quadratic)}
