"""Public OSM data -> directed road multigraph -> reproducible small study areas."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

import networkx as nx

# Small area near BMSCE. These bounds do not identify an actual ambulance station.
BBOX = (77.560, 12.936, 77.570, 12.946)  # west, south, east, north
URL = 'https://api.openstreetmap.org/api/0.6/map?bbox=' + ','.join(map(str, BBOX))
SPEEDS = {'motorway': 60, 'trunk': 50, 'primary': 40, 'secondary': 35,
          'tertiary': 30, 'residential': 25, 'unclassified': 25,
          'living_street': 15, 'service': 15}
SPEEDS.update({k + '_link': v for k, v in list(SPEEDS.items()) if k in {'motorway','trunk','primary','secondary','tertiary'}})


def distance(a, b):
    """Great-circle distance in metres; a and b are (longitude, latitude)."""
    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371000 * 2 * math.asin(min(1, math.sqrt(h)))


def download(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return  # Do not silently replace the dataset used in earlier experiments.
    request = urllib.request.Request(URL, headers={'User-Agent':'AmbulanceRoutingAcademicPrototype/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()
    ET.fromstring(content)  # Verify XML before saving.
    path.write_bytes(content)


def parse_osm(path):
    """Conservative drivable-road subset; not a complete legal navigation engine."""
    root = ET.parse(path).getroot()
    nodes = {}
    for item in root.findall('node'):
        x, y = float(item.attrib['lon']), float(item.attrib['lat'])
        if BBOX[0] <= x <= BBOX[2] and BBOX[1] <= y <= BBOX[3]:
            tags = {t.attrib['k']:t.attrib['v'] for t in item.findall('tag')}
            nodes[item.attrib['id']] = {'x':x, 'y':y, 'signal':tags.get('highway') == 'traffic_signals'}
    raw = nx.MultiDiGraph()
    endpoints = set()
    for way in root.findall('way'):
        tags = {t.attrib['k']:t.attrib['v'] for t in way.findall('tag')}
        road = tags.get('highway')
        if road not in SPEEDS or any(tags.get(k) in {'no','private'} for k in ('access','vehicle','motor_vehicle')):
            continue
        refs = [n.attrib['ref'] for n in way.findall('nd')]
        if len(refs) < 2:
            continue
        endpoints.update((refs[0], refs[-1]))
        speed = SPEEDS[road]  # Explicit class-based simulation assumption, not measured/maxspeed.
        oneway = tags.get('oneway', 'yes' if tags.get('junction') == 'roundabout' or road == 'motorway' else 'no')
        for index, (u, v) in enumerate(zip(refs, refs[1:])):
            if u not in nodes or v not in nodes or u == v:
                continue
            length = distance((nodes[u]['x'], nodes[u]['y']), (nodes[v]['x'], nodes[v]['y']))
            if length <= 0:
                continue
            directions = [(u,v)] if oneway in {'yes','1','true'} else [(v,u)] if oneway == '-1' else [(u,v),(v,u)]
            for a,b in directions:
                raw.add_node(a, **nodes[a]); raw.add_node(b, **nodes[b])
                eid = f"{way.attrib['id']}:{index}:{a}:{b}"
                raw.add_edge(a,b,key=eid,edge_id=eid,length_m=length,base_time_s=length/(speed/3.6),
                             geometry=[[nodes[a]['x'],nodes[a]['y']],[nodes[b]['x'],nodes[b]['y']]])
    if raw.number_of_edges() == 0:
        raise ValueError('No usable drivable roads in the input')
    # Contract only unambiguous degree-two chains; retain geometry and total time.
    undirected = nx.Graph(raw)
    keep = {v for v in raw if undirected.degree(v) != 2 or v in endpoints or raw.nodes[v]['signal']}
    # Preserve nodes with parallel choices to prevent silently losing roads.
    keep.update(v for v in raw if any(len(raw[v][w]) > 1 for w in raw.successors(v)))
    for component in nx.connected_components(undirected):
        if not (set(component) & keep):
            keep.add(min(component))
    graph = nx.MultiDiGraph()
    for v in sorted(keep):
        graph.add_node(v, **raw.nodes[v])
    for start in sorted(keep):
        for _, first, key, edge in raw.out_edges(start, keys=True, data=True):
            previous, current = start, first
            parts = [edge]; seen = {start}
            while current not in keep:
                if current in seen:
                    break
                seen.add(current)
                choices = [(w,k,d) for _,w,k,d in raw.out_edges(current,keys=True,data=True) if w != previous]
                if len(choices) != 1:
                    break
                following, _, data = choices[0]
                parts.append(data); previous,current = current,following
            if current not in keep or current == start:
                continue
            geometry = list(parts[0]['geometry'])
            for part in parts[1:]: geometry.extend(part['geometry'][1:])
            eid = edge['edge_id']
            graph.add_edge(start,current,key=eid,edge_id=eid,
                           length_m=sum(p['length_m'] for p in parts),
                           base_time_s=sum(p['base_time_s'] for p in parts), geometry=geometry)
    # A strongly connected component permits directed travel in both directions via paths.
    component = max(nx.strongly_connected_components(graph),key=len)
    graph = graph.subgraph(component).copy()
    validate_graph(graph)
    return graph


def save_graph(graph, path, source_path=None):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    metadata = {'source':'OpenStreetMap contributors', 'source_url':URL, 'license':'ODbL 1.0',
                'license_url':'https://www.openstreetmap.org/copyright', 'bbox':BBOX,
                'processed_at_utc':datetime.now(timezone.utc).isoformat(),
                'speed_assumptions_kph':SPEEDS, 'traffic':'synthetic; no observed traffic data',
                'raw_sha256':hashlib.sha256(Path(source_path).read_bytes()).hexdigest() if source_path else None}
    payload = {'metadata':metadata, 'nodes':[{'id':n,**d} for n,d in graph.nodes(data=True)],
               'edges':[{'u':u,'v':v,'key':k,**d} for u,v,k,d in graph.edges(keys=True,data=True)]}
    path.write_text(json.dumps(payload,indent=2),encoding='utf-8')


def load_graph(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    graph = nx.MultiDiGraph()
    for row in payload['nodes']:
        row = dict(row); node = row.pop('id'); graph.add_node(node,**row)
    ids = set()
    for row in payload['edges']:
        row = dict(row); u,v,k = row.pop('u'),row.pop('v'),row.pop('key')
        if u not in graph or v not in graph or k in ids:
            raise ValueError('Missing endpoint or duplicate edge ID')
        ids.add(k); graph.add_edge(u,v,key=k,**row)
    validate_graph(graph)
    return graph


def validate_graph(graph):
    for n,d in graph.nodes(data=True):
        if not all(math.isfinite(d[k]) for k in ('x','y')):
            raise ValueError(f'Invalid coordinates for {n}')
    for u,v,k,d in graph.edges(keys=True,data=True):
        if u == v or d['edge_id'] != k:
            raise ValueError('Self-loop or inconsistent edge ID')
        for name in ('length_m','base_time_s'):
            if not math.isfinite(d[name]) or d[name] <= 0: raise ValueError(f'Invalid {name}')


def study_patch(graph, count):
    """Nested geographically central connected patches. Actual sizes are recorded."""
    if count < 4: raise ValueError('Use at least four nodes')
    center = ((BBOX[0]+BBOX[2])/2,(BBOX[1]+BBOX[3])/2)
    anchor = min(graph,key=lambda n: distance(center,(graph.nodes[n]['x'],graph.nodes[n]['y'])))
    order = [anchor] + [v for _,v in nx.bfs_edges(nx.Graph(graph),anchor,sort_neighbors=sorted)]
    return graph.subgraph(order[:count]).copy()


def toy_graph():
    """Five-road directed fixture, small enough to enumerate all 32 selections."""
    g = nx.MultiDiGraph()
    for n,(x,y) in {'S':(0,0),'A':(.001,0),'B':(0,.001),'T':(.001,.001)}.items():
        g.add_node(n,x=x,y=y,signal=False)
    for eid,u,v,t in [('sa','S','A',120),('at','A','T',120),('sb','S','B',150),('bt','B','T',150),('ab','A','B',80)]:
        g.add_edge(u,v,key=eid,edge_id=eid,length_m=1000,base_time_s=t,time_s=float(t),
                   geometry=[[g.nodes[u]['x'],g.nodes[u]['y']],[g.nodes[v]['x'],g.nodes[v]['y']]])
    return g
