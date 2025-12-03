import argparse
import time
from src.geometry import generate_random_cities
from src.parallel_engine import ParallelManager
from src.tsp_solver import GeneticAlgorithm

def main():
    parser = argparse.ArgumentParser(description="Parallel Evolutionary TSP Solver")
    parser.add_argument("--cities", type=int, default=100, help="Number of cities")
    parser.add_argument("--islands", type=int, default=4, help="Number of parallel islands (cores)")
    parser.add_argument("--generations", type=int, default=100, help="Generations per island")
    parser.add_argument("--serial", action="store_true", help="Run in serial mode (single core)")
    
    args = parser.parse_args()
    
    print(f"Generating {args.cities} random cities...")
    points = generate_random_cities(args.cities, seed=42)
    
    start_time = time.time()
    
    if args.serial:
        print("Running in SERIAL mode...")
        ga = GeneticAlgorithm(points, pop_size=50)
        # Run for equivalent total generations (islands * generations) to be fair? 
        # Or just same wall time? Usually same generations per run.
        # Let's run single GA for islands * generations to match total compute?
        # No, usually we compare wall time for same quality or quality for same time.
        # Let's run standard GA for same generations as one island but it's just one process.
        
        best_fitness_history = []
        for gen in range(args.generations):
            best = ga.evolve()
            if gen % 50 == 0:
                ga.run_local_search()
        
        best_tour = ga.population[0]
        print(f"Final Distance: {best_tour.fitness:.2f}")
        
    else:
        print(f"Running in PARALLEL mode with {args.islands} islands...")
        manager = ParallelManager(points, num_islands=args.islands, generations=args.generations)
        result = manager.run()
        island_id, best_tour, history = result
        print(f"Best Solution found by Island {island_id}")
        print(f"Final Distance: {best_tour.fitness:.2f}")

    elapsed = time.time() - start_time
    print(f"Total Execution Time: {elapsed:.4f} seconds")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
