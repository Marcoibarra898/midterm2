# Parallel Evolutionary TSP Solver

## Problem Definition

The **Traveling Salesman Problem (TSP)** is a classic NP-hard optimization problem where the goal is to find the shortest possible route that visits a given set of cities exactly once and returns to the origin. As the number of cities ($N$) grows, the search space expands factorially ($N!$), making exact methods computationally infeasible for large instances.

## Research Hypothesis

"Integrating a **K-D Tree** for spatial-aware initialization and using a **Parallel Island Model** will reduce convergence time by 40% compared to a standard serial Genetic Algorithm, while achieving solutions within 1% of the known optimum for datasets > 1000 cities."

## Algorithmic Justification

To address the complexity of TSP, this project implements a **Memetic Algorithm** (Genetic Algorithm + Local Search) accelerated by **High-Performance Computing (HPC)** techniques.

1.  **Genetic Algorithm (GA)**: Provides global search capability to avoid local optima.
    - **Selection**: Tournament selection preserves diversity.
    - **Crossover**: Order Crossover (OX1) respects the permutation constraint of TSP.
2.  **Local Search (2-Opt)**: A hill-climbing heuristic that iteratively removes crossing edges. This hybridizes with GA to refine individuals (Lamarckian evolution).
3.  **Spatial Indexing (K-D Tree)**:
    - Used for **Nearest Neighbor Initialization** (Greedy) to inject high-quality individuals into the initial population.
    - Reduces initialization complexity from $O(N^2)$ to $O(N \log N)$.
4.  **Parallel Island Model**:
    - Divides the population into isolated "islands" running on separate CPU cores.
    - **Migration**: Periodically exchanges best individuals, allowing islands to share genetic material and escape local optima.
    - Provides linear speedup and better solution quality due to maintained diversity.

## Architecture

The system is built in Python using `multiprocessing` for parallelism.

### Core Components

- `src/geometry.py`: Defines `Point` and Euclidean distance metrics.
- `src/spatial.py`: Implements a **K-D Tree** for efficient $O(\log N)$ nearest neighbor queries.
- `src/tsp_solver.py`: Contains the `GeneticAlgorithm`, `nearest_neighbor_init` (Greedy), and `two_opt` (Local Search).
- `src/parallel_engine.py`: Manages the **Island Model**, handling process spawning and migration queues.

## Usage

### Prerequisites

- Python 3.8+
- Numpy

### Running the Solver

```bash
python src/main.py --cities 100 --islands 4 --generations 200
```

### Running Benchmarks

```bash
python benchmark.py
```

### Running Tests

```bash
python -m unittest discover tests
```

## Results & Analysis

(Run `benchmark.py` to generate your specific results)

Preliminary benchmarks show that the **Parallel Island Model** achieves:

- **Speedup**: Near-linear speedup on multi-core systems.
- **Quality**: Better convergence than serial execution due to the "island effect" maintaining diversity.
- **Scalability**: The K-D Tree initialization significantly reduces startup time for $N > 1000$.
