# QUBO-Based Hybrid Ambulance Route Optimization

An academic research prototype that compares classical shortest-path algorithms with QUBO-based simulated annealing and a candidate-based hybrid method for ambulance routing on a small public road network near BMS College of Engineering, Bengaluru.

## Problem Statement

The shortest-distance road route may not provide the shortest ambulance travel time. Congestion, traffic signals, accidents and road closures can affect the preferred route.

This project models a road network as a directed graph and seeks a valid source-to-destination route with minimum estimated travel time under simulated traffic conditions. It evaluates a QUBO formulation against classical shortest-path methods.

## Objectives

- Use a publicly available road-network dataset.
- Simulate congestion, signal delays, accidents and closures.
- Formulate route selection using binary variables and QUBO penalties.
- Compare simulated annealing and a candidate-based hybrid with Dijkstra, A* and Bellman–Ford.
- Evaluate route feasibility, travel time, optimality gap and computation time.
- Demonstrate rerouting after a simulated closure.
- Provide reproducible code, tests and experiment outputs.

## Current Implementation

### Core Routing — Professor's Phase I Scope

Implemented features include:

- Public OpenStreetMap road data.
- Directed road graphs with parallel-road support.
- Traffic snapshots with positive travel-time weights.
- Edge-based routing QUBO.
- Three classical shortest-path baselines.
- Raw simulated annealing.
- Candidate-subgraph hybrid optimization.
- Route validation and comparison metrics.
- Route maps and experiment plots.
- Rerouting and penalty-sensitivity experiments.
- A separate small simulator-based QAOA demonstration.

The implementation is a research prototype. Completing the code does not establish research novelty, publication readiness or practical emergency-service effectiveness.

### Exploratory Extensions

The existing source and command names use `phase2` and `phase3`:

| Extension | Features |
|---|---|
| `phase2.py` | Assignment of available ambulances to simultaneous synthetic emergencies and hospital-capacity slots |
| `phase3.py` | Independent traffic-scenario comparisons with fixed and adaptively selected QUBO penalties |

These names refer to implementation stages, not confirmation that the professor approved or completed additional academic phases.

The fleet extensions use synthetic ambulance, emergency and hospital conditions. Road costs are computed classically before constructing a separate assignment QUBO.

The scenario extension does not carry vehicle positions or hospital occupancy forward between scenarios. It is not a continuous fleet-motion or chronological rolling-horizon simulation.

## Technology Stack

| Technology | Use |
|---|---|
| Python 3.12 | Main implementation |
| NetworkX | Road graphs and classical graph algorithms |
| NumPy | Numerical operations and toy statevector simulation |
| SciPy | Classical numerical optimization |
| dimod | Binary quadratic model representation |
| dwave-samplers | Local classical simulated annealing |
| Matplotlib | Maps and comparison plots |
| unittest | Automated implementation tests |

The simulated annealing solver runs locally on a classical computer. No quantum hardware, D-Wave account or cloud token is required.

## Dataset

**Source:** OpenStreetMap  
**Location:** Small area near BMSCE, Bengaluru, India  
**Attribution:** © OpenStreetMap contributors  
**Database license:** Open Database License (ODbL) 1.0

Website: https://www.openstreetmap.org/

Download URL:

https://api.openstreetmap.org/api/0.6/map?bbox=77.560,12.936,77.570,12.946

The bounding box is specified as west longitude, south latitude, east longitude and north latitude.

| Property | Value |
|---|---|
| Bounding box | 77.560, 12.936, 77.570, 12.946 |
| Processed graph | 375 nodes and 1,024 directed arcs |
| Representation | NetworkX MultiDiGraph |
| Default core experiment patches | 12, 24 and 40 nodes |
| Raw dataset | `data/bmsce_area.osm` |
| Processed dataset | `data/road_graph.json` |

Nodes represent retained road-network points; not every node is necessarily an intersection. Opposite travel directions count as separate arcs.

### Real Data and Simulated Inputs

Real information includes road geometry, coordinates, connectivity and available OSM tags.

Simulated inputs include road-class speeds, congestion, incident delays, closures, ambulance positions, emergency severity/deadlines and hospital conditions.

No live traffic feed, ambulance GPS records, patient records or measured emergency response times are used.

### Preprocessing

1. Read the downloaded OSM XML.
2. Retain supported drivable road classes.
3. Filter explicit private/no-access roads.
4. Interpret basic one-way and roundabout direction.
5. Calculate road lengths from coordinates.
6. Estimate base travel times using assumed road-class speeds.
7. Simplify unambiguous road chains while preserving geometry.
8. Retain the largest strongly connected component.
9. Save nodes, edges and provenance metadata to JSON.

Complete turn restrictions, conditional access rules and emergency exemptions are not modeled.

Normal experiments load the bundled JSON without downloading the map again. Its metadata records the source URL, bounding box, processing timestamp, speed assumptions and raw-file SHA-256 fingerprint.

## File Descriptions

| File | Responsibility |
|---|---|
| `main.py` | Parses commands and starts the requested workflow |
| `src/__init__.py` | Defines the source package |
| `src/data.py` | Downloads, processes, saves and loads road data; creates study patches and the toy graph |
| `src/traffic.py` | Applies congestion, signal delays, accidents and road closures |
| `src/routes.py` | Classical baselines, route decoding, validation and metrics |
| `src/qubo.py` | Constructs the routing QUBO |
| `src/solvers.py` | Runs baselines, raw SA and the hybrid method |
| `src/experiments.py` | Organizes core experiments and saves results and manifests |
| `src/plotting.py` | Produces route maps and comparison plots |
| `src/quantum_demo.py` | Implements the separate toy QAOA simulation |
| `src/phase2.py` | Implements the fleet and hospital-slot assignment extension |
| `src/phase3.py` | Implements adaptive penalty selection and scenario comparisons |
| `tests/test_research.py` | Tests core routing, QUBO and validation behavior |
| `tests/test_phase2.py` | Tests assignment-model behavior |
| `tests/test_phase3.py` | Tests adaptive-penalty extension behavior |
| `data/bmsce_area.osm` | Raw OpenStreetMap extract |
| `data/road_graph.json` | Processed graph and metadata |
| `data/SOURCE_DATA.md` | Data provenance and preprocessing notes |
| `data/LICENSE.md` | Dataset attribution and license notice |
| `requirements.txt` | Pinned direct dependencies |
| `requirements-lock.txt` | Recorded dependency versions |
| `.gitignore` | Excludes local environments, teaching files and generated outputs |

A notebook is not required to execute the implementation.

## Traffic Model

Base travel time is calculated as:

`base_time_s = length_m / (speed_kph / 3.6)`

The traffic module applies:

`time_s = base_time_s × congestion_multiplier + delay`

- Congestion varies deterministically by road, scenario and selected hour.
- Signal-tagged destination nodes add 20 seconds.
- Selected accident roads receive 180 seconds of additional delay.
- Selected blocked directed roads are removed from the graph.

Traffic costs remain fixed during each route calculation. They are simulated snapshots, not live traffic predictions.

## QUBO Formulation

For each directed road e, define a binary variable x_e:

- x_e = 1: select the road.
- x_e = 0: do not select the road.

Let c_e be travel time divided by the maximum edge travel time. Let b_v equal 1 at the source, -1 at the destination and 0 elsewhere.

The objective is:

```math
E(x)=\sum_e c_e x_e
+P\sum_v\left(\sum_{e\in out(v)}x_e-\sum_{e\in in(v)}x_e-b_v\right)^2
+P\sum_v\left(\sum_{e<f\in out(v)}x_ex_f+\sum_{e<f\in in(v)}x_ex_f\right).
```

The terms represent:

1. Normalized travel cost.
2. Flow-balance penalties.
3. Penalties for selecting multiple incoming or outgoing roads.

A feasible fewest-edge reference route supplies a penalty bound:

`P = penalty_factor × normalized_reference_cost + 1`

The default penalty factor is 2. Binary squares are expanded using x² = x.

Positive travel costs discourage extra cycles at the exact optimum. Since heuristic samples may still be invalid, every sampled selection is checked for a single valid route without extra roads, branches or cycles.

## Methods Compared

| Method | Description |
|---|---|
| Dijkstra | Exact shortest-time reference for positive snapshot costs |
| A* | Classical shortest path using a conservative geographic heuristic |
| Bellman–Ford | Additional classical shortest-path baseline |
| Raw SA | Simulated annealing on the full supplied routing graph |
| Hybrid | Candidate generation, reduced QUBO, seeded annealing and candidate retention |

### Hybrid Procedure

1. Generate up to four shortest-distance simple paths.
2. Add a fewest-edge path.
3. Build a reduced graph from their road connections.
4. Convert candidate routes into binary initial states.
5. Run simulated annealing on the reduced QUBO.
6. Validate the samples.
7. Return the best sampled route only if it improves the best candidate; otherwise retain the candidate.

Candidate pruning can exclude the globally fastest route.

The result fields `candidate_retained`, `candidate_time_s`, `raw_sa_time_s` and `improved_over_candidate` distinguish classical candidate quality from annealing improvement.

## Setup and Execution

Run commands from the folder containing `main.py`.

### Windows Command Prompt

Create a Python 3.12 environment if this project does not already have one:

```bat
py -3.12 -m venv .venv
```

Install dependencies:

```bat
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run tests:

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run the toy example:

```bat
.venv\Scripts\python.exe main.py demo --out outputs/toy_01
```

Run the public-road example:

```bat
.venv\Scripts\python.exe main.py demo --public --out outputs/public_01
```

Run a small comparison experiment:

```bat
.venv\Scripts\python.exe main.py experiment --quick --out outputs/quick_01
```

Run the standard core experiment:

```bat
.venv\Scripts\python.exe main.py experiment --out outputs/study_01
```

Run additional demonstrations:

```bat
.venv\Scripts\python.exe main.py reroute --out outputs/reroute_01
.venv\Scripts\python.exe main.py penalty-study --out outputs/penalties_01
.venv\Scripts\python.exe main.py quantum-demo --out outputs/quantum_01
.venv\Scripts\python.exe main.py phase2-demo --out outputs/fleet_01
.venv\Scripts\python.exe main.py phase3-demo --out outputs/adaptive_01
```

Use `python main.py --help` with the configured project interpreter to see available commands.

If using an existing environment elsewhere, replace `.venv\Scripts\python.exe` with its actual interpreter path. Use a new output directory for each run; the project protects nonempty result directories from overwriting.

## Experiment Design

The default core study evaluates:

- Three graph sizes: 12, 24 and 40 nodes.
- Three source–destination pairs per size.
- Four scenarios: normal, peak, accident and roadblock.
- Three seeds for each stochastic method.

This produces 36 routing tasks and 324 result rows, including failures. Classical methods run once per task; SA and hybrid each run once per seed.

Default standard-study sampling uses 64 reads and 1,000 sweeps. The quick study uses a smaller predefined configuration. Equal sampling budgets do not imply equal computation time.

## Evaluation Metrics

| Metric | Meaning |
|---|---|
| Travel time | Estimated ambulance journey time |
| Distance | Total route length |
| Runtime | Computer time required by the solver |
| Optimality gap | Percentage above the same-graph Dijkstra optimum |
| Raw feasible fraction | Fraction of annealing samples forming valid routes |
| QUBO size | Number of variables and pairwise interactions |
| Candidate improvement | Whether annealing improves the best hybrid candidate |

`gap_pct = 100 × (method_time − optimal_time) / optimal_time`

Status values distinguish:

- `ok`: a valid solution was returned.
- `no_route`: the graph does not connect the endpoints.
- `solver_failed`: a route exists but the solver returned no valid solution.

Failures must be reported alongside successful-route gaps.

Solver runtime includes validation, reachability checks, preprocessing, QUBO construction, sampling and decoding. Shared dataset loading, traffic construction, plotting and file writing are excluded.

## Generated Outputs

| Workflow | Main outputs |
|---|---|
| Demo | `results.csv`, `routes.png`, `qubo_model.json`, `manifest.json` |
| Core study | `raw_results.csv`, `tasks.json`, comparison plots, `SUMMARY.md`, `manifest.json` |
| Rerouting | `reroute.json`, `reroute.png`, `manifest.json` |
| Penalty study | `penalty_results.csv`, `manifest.json` |
| Quantum demonstration | `quantum_demo.json`, `manifest.json` |
| Fleet extensions | Assignment/event results, scenario information, plots and summaries |

Core manifests record settings, package versions and code/data fingerprints for reproducibility.

Generated outputs are excluded from Git by default. Reviewed experiment evidence can be deliberately added to a separate tracked `results/` folder with its configuration and run environment.

## Validation and Limitations

The test suite checks selected properties of routing, QUBO construction, assignment feasibility and penalty selection. Passing tests does not demonstrate real-world effectiveness or algorithmic superiority.

Current limitations include:

- One small geographic area.
- Synthetic traffic and emergency-service inputs.
- Frozen travel-time snapshots.
- No explicit road-risk objective or live hospital data.
- Possible loss of good routes through candidate pruning.
- Limited stochastic repetitions.
- Small-instance exact enumeration for fleet comparisons.
- A toy quantum simulation separate from public-road optimization.

Classical shortest-path algorithms already solve the core fixed-cost routing problem exactly. No quantum speedup or guaranteed advantage is claimed.

Remaining research work includes verified related-work analysis, a standalone candidate-only baseline, controlled reduction/seeding comparisons, interpretation of measured results, final paper figures and supervisor review.

## Data Attribution

The map databases contain information from OpenStreetMap, © OpenStreetMap contributors, available under ODbL 1.0.

https://www.openstreetmap.org/copyright

Retain `data/LICENSE.md` and `data/SOURCE_DATA.md` when redistributing the dataset. The dataset license does not automatically establish a source-code license.