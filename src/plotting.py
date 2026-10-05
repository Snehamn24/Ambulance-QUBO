"""Offline route geometry and basic runtime/quality plots; no map tiles required."""
from collections import defaultdict
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.routes import edge_lookup


def route_map(graph,source,target,results,path,title):
    rows=(len(results)+2)//3
    fig,axes=plt.subplots(rows,3,figsize=(13,4*rows),squeeze=False)
    lookup=edge_lookup(graph)
    for ax,result in zip(axes.flat,results):
        for _,_,_,d in graph.edges(keys=True,data=True):
            x,y=zip(*d['geometry']); ax.plot(x,y,color='#d5d9dd',lw=1,zorder=1)
        for k in result.get('path',[]):
            u,v,d=lookup[k]; x,y=zip(*d['geometry']); ax.plot(x,y,color='#087f8c',lw=2.5,zorder=3)
            if len(x)>1: ax.annotate('',xy=(x[-1],y[-1]),xytext=(x[-2],y[-2]),arrowprops={'arrowstyle':'->','color':'#087f8c'})
        for node,label,color in ((source,'S','#c53030'),(target,'T','#253494')):
            d=graph.nodes[node]; ax.scatter(d['x'],d['y'],color=color,zorder=4); ax.annotate(label,(d['x'],d['y']),fontsize=11)
        time=result.get('travel_time_s')
        ax.set_title(f"{result['method']} | {time/60:.2f} min" if time is not None else f"{result['method']} | {result['status']}")
        ax.set_aspect('equal'); ax.tick_params(labelsize=7); ax.ticklabel_format(useOffset=False)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    for ax in list(axes.flat)[len(results):]: ax.axis('off')
    attribution = 'Synthetic correctness fixture; coordinates are illustrative.' if title.startswith('Toy') else 'Road geometry: © OpenStreetMap contributors, ODbL. Speeds and traffic simulated.'
    fig.suptitle(title); fig.text(.5,.01,attribution,ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94]); fig.savefig(path,dpi=160); plt.close(fig)


def comparisons(rows,folder):
    folder=Path(folder)
    grouped=defaultdict(list)
    for row in rows: grouped[(row['method'],row['actual_nodes'])].append(row)
    fig,axes=plt.subplots(1,3,figsize=(14,4.3))
    for method in sorted({r['method'] for r in rows}):
        xs=sorted(n for m,n in grouped if m==method)
        medians=[statistics.median(r['runtime_ms'] for r in grouped[method,n]) for n in xs]
        gaps=[]; successes=[]
        for n in xs:
            available=[r for r in grouped[method,n] if r['status']!='no_route']
            valid=[r['gap_pct'] for r in available if r.get('gap_pct') is not None]
            gaps.append(statistics.median(valid) if valid else float('nan'))
            successes.append(sum(r['status']=='ok' for r in available)/len(available) if available else float('nan'))
        axes[0].plot(xs,medians,'o-',label=method)
        axes[1].plot(xs,gaps,'o-',label=method)
        axes[2].plot(xs,successes,'o-',label=method)
    axes[0].set_yscale('log'); axes[0].set_ylabel('Median solver runtime (ms; log scale)')
    axes[1].set_ylabel('Median gap to Dijkstra (%) among valid runs')
    axes[2].set_ylabel('Valid-result fraction among reachable tasks'); axes[2].set_ylim(-.05,1.05)
    for ax in axes: ax.set_xlabel('Actual nodes in study patch'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('One public road area, synthetic traffic — failures retained in result table')
    fig.tight_layout(); fig.savefig(folder/'runtime_quality_scalability.png',dpi=160); plt.close(fig)
