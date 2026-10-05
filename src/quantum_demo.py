"""Tiny p=1 QAOA statevector simulator; no cloud account or quantum hardware.

The amplitudes are computed directly with NumPy. This educational component is
NOT simulated annealing and is NOT part of the city-scale timing comparison.
"""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from src.data import toy_graph
from src.qubo import build_qubo
from src.routes import decode, metrics


def statevector(energies,gamma,beta):
    count=len(energies); n=int(np.log2(count))
    state=np.ones(count,dtype=complex)/np.sqrt(count)
    state *= np.exp(-1j*gamma*energies)
    # exp(-i beta X) on every qubit; little-endian bit positions.
    for bit in range(n):
        stride=1 << bit
        for start in range(0,count,2*stride):
            left=state[start:start+stride].copy(); right=state[start+stride:start+2*stride].copy()
            state[start:start+stride]=np.cos(beta)*left-1j*np.sin(beta)*right
            state[start+stride:start+2*stride]=np.cos(beta)*right-1j*np.sin(beta)*left
    return state


def run_demo(output,seed=2026):
    start=perf_counter(); graph=toy_graph(); bqm,_=build_qubo(graph,'S','T')
    variables=list(bqm.variables); n=len(variables)
    if n > 12: raise ValueError('Statevector demo is restricted to at most 12 qubits')
    samples=[{v:(index >> bit)&1 for bit,v in enumerate(variables)} for index in range(1<<n)]
    energies=np.array([bqm.energy(x) for x in samples])
    scaled=(energies-energies.min())/max(1.,np.ptp(energies))
    def expectation(theta):
        return float(np.dot(abs(statevector(scaled,*theta))**2,scaled))
    rng=np.random.default_rng(seed)
    results=[minimize(expectation,rng.uniform(0,np.pi,size=2),method='Nelder-Mead',options={'maxiter':120}) for _ in range(4)]
    result=min(results,key=lambda r:r.fun)
    probabilities=abs(statevector(scaled,*result.x))**2
    feasible=[]
    for i,sample in enumerate(samples):
        route=decode(graph,'S','T',[v for v,b in sample.items() if b])
        if route is not None: feasible.append((i,route))
    minimum=min(metrics(graph,p)['travel_time_s'] for _,p in feasible)
    optimal=[i for i,p in feasible if abs(metrics(graph,p)['travel_time_s']-minimum)<1e-8]
    payload={'component':'p=1 QAOA NumPy statevector; toy graph only','seed':seed,
             'qubits':n,'gamma':float(result.x[0]),'beta':float(result.x[1]),
             'optimizer_success':bool(result.success),'iterations':int(result.nit),
             'feasible_probability':float(sum(probabilities[i] for i,_ in feasible)),
             'optimal_route_probability':float(sum(probabilities[i] for i in optimal)),
             'uniform_feasible_probability':len(feasible)/(1<<n),
             'uniform_optimal_probability':len(optimal)/(1<<n),
             'exact_optimal_travel_time_s':minimum,'runtime_ms':(perf_counter()-start)*1000,
             'interpretation':'Exact probabilities, not shot counts. No quantum advantage claimed.'}
    Path(output).write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return payload
