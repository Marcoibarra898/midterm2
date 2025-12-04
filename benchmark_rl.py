import argparse
import time
import numpy as np
from src.geometry import generate_random_cities, load_cities_from_csv, calculate_tour_distance
from src.tsp_solver import GeneticAlgorithm
from src.rl_solver import QLearningTSPSolver

def run_ga(points, generations=100):
    print(f"--- Running Genetic Algorithm ({generations} generations) ---")
    start_time = time.time()
    ga = GeneticAlgorithm(points, pop_size=50)
    for gen in range(generations):
        ga.evolve()
        if gen % 50 == 0:
            ga.run_local_search()
    
    best_tour = ga.population[0]
    elapsed = time.time() - start_time
    print(f"GA Completed in {elapsed:.4f}s")
    print(f"GA Distance: {best_tour.fitness:.2f}")
    return best_tour.fitness, elapsed

def run_rl(points, episodes=1000):
    print(f"--- Running Q-Learning ({episodes} episodes) ---")
    start_time = time.time()
    solver = QLearningTSPSolver(points, alpha=0.1, gamma=0.99, epsilon=0.1)
    solver.train(episodes=episodes)
    
    tour = solver.get_solution()
    dist = calculate_tour_distance(tour)
    elapsed = time.time() - start_time
    print(f"RL Completed in {elapsed:.4f}s")
    print(f"RL Distance: {dist:.2f}")
    return dist, elapsed

def main():
    parser = argparse.ArgumentParser(description="TSP Benchmark: GA vs RL")
    parser.add_argument("--cities", type=int, default=20, help="Number of cities")
    parser.add_argument("--file", type=str, help="Path to CSV file")
    parser.add_argument("--episodes", type=int, default=1000, help="RL Episodes")
    parser.add_argument("--generations", type=int, default=100, help="GA Generations")
    
    args = parser.parse_args()
    
    if args.file:
        print(f"Loading cities from {args.file}...")
        points = load_cities_from_csv(args.file)
    else:
        print(f"Generating {args.cities} random cities...")
        points = generate_random_cities(args.cities, seed=42)
        
    print(f"Problem Size: {len(points)} cities")
    
    # Run GA
    ga_dist, ga_time = run_ga(points, generations=args.generations)
    
    # Run RL
    rl_dist, rl_time = run_rl(points, episodes=args.episodes)
    
    print("\n--- Comparison Results ---")
    print(f"{'Method':<10} | {'Distance':<10} | {'Time (s)':<10}")
    print("-" * 36)
    print(f"{'GA':<10} | {ga_dist:<10.2f} | {ga_time:<10.4f}")
    print(f"{'RL':<10} | {rl_dist:<10.2f} | {rl_time:<10.4f}")
    
    gap = ((rl_dist - ga_dist) / ga_dist) * 100
    print(f"\nGap (RL vs GA): {gap:+.2f}% (Positive means RL is worse)")

if __name__ == "__main__":
    main()
