import time
import csv
from src.geometry import generate_random_cities
from src.parallel_engine import ParallelManager
from src.tsp_solver import GeneticAlgorithm

def run_benchmark():
    city_counts = [50, 100, 200, 500]
    core_counts = [1, 2, 4, 8]
    generations = 100
    
    results = []
    
    print("Starting Benchmark Suite...")
    print("Cities | Cores | Time (s) | Distance | Speedup")
    print("-" * 45)
    
    for n in city_counts:
        points = generate_random_cities(n, seed=42)
        
        # Baseline (Serial / 1 Core)
        start_serial = time.time()
        # Simulate 1 core using ParallelManager with 1 island for consistency
        manager_serial = ParallelManager(points, num_islands=1, generations=generations)
        res_serial = manager_serial.run()
        time_serial = time.time() - start_serial
        dist_serial = res_serial[1].fitness
        
        results.append({
            "cities": n,
            "cores": 1,
            "time": time_serial,
            "distance": dist_serial,
            "speedup": 1.0
        })
        print(f"{n:<6} | 1     | {time_serial:<8.4f} | {dist_serial:<8.2f} | 1.00")
        
        for cores in core_counts:
            if cores == 1: continue
            
            start_par = time.time()
            manager = ParallelManager(points, num_islands=cores, generations=generations)
            res = manager.run()
            time_par = time.time() - start_par
            dist_par = res[1].fitness
            
            speedup = time_serial / time_par
            
            results.append({
                "cities": n,
                "cores": cores,
                "time": time_par,
                "distance": dist_par,
                "speedup": speedup
            })
            print(f"{n:<6} | {cores:<5} | {time_par:<8.4f} | {dist_par:<8.2f} | {speedup:.2f}")

    # Save to CSV
    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cities", "cores", "time", "distance", "speedup"])
        writer.writeheader()
        writer.writerows(results)
    
    print("\nBenchmark completed. Results saved to benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()
