import multiprocessing
import time
from typing import List
from src.geometry import Point
from src.tsp_solver import GeneticAlgorithm, Individual

def run_island(points: List[Point], generations: int, island_id: int, migration_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue):
    """
    Runs a single GA island.
    """
    ga = GeneticAlgorithm(points, pop_size=50, mutation_rate=0.1)
    best_fitness_history = []
    
    for gen in range(generations):
        best_ind = ga.evolve()
        
        # Periodic Local Search (Memetic Algorithm)
        if gen % 50 == 0:
            ga.run_local_search()
            best_ind = ga.population[0] # Update after LS

        # Log Progress
        if gen % 10 == 0:
            print(f"[Island {island_id}] Gen {gen}/{generations} | Best: {best_ind.fitness:.2f}", flush=True)

        best_fitness_history.append(best_ind.fitness)
        
        # Migration: Every 20 generations
        if gen % 20 == 0:
            # Send best to manager/other islands
            migration_queue.put((island_id, best_ind))
            
            # Receive migrants
            import queue
            try:
                # Try to get a few migrants (limit to avoid processing entire history)
                for _ in range(2): 
                    if not migration_queue.empty():
                        sender_id, migrant = migration_queue.get_nowait()
                        if sender_id != island_id:
                            # Replace worst individual with migrant
                            # Assuming population is sorted (best first)
                            ga.population[-1] = migrant
                            ga.population.sort()
            except queue.Empty:
                pass

    result_queue.put((island_id, ga.population[0], best_fitness_history))

class ParallelManager:
    def __init__(self, points: List[Point], num_islands: int = 4, generations: int = 100):
        self.points = points
        self.num_islands = num_islands
        self.generations = generations

    def run(self):
        migration_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()
        
        processes = []
        for i in range(self.num_islands):
            p = multiprocessing.Process(
                target=run_island,
                args=(self.points, self.generations, i, migration_queue, result_queue)
            )
            processes.append(p)
            p.start()
            
        # Collect results
        results = []
        for _ in range(self.num_islands):
            results.append(result_queue.get())
            
        for p in processes:
            p.join()
            
        # Find global best
        global_best = min(results, key=lambda x: x[1].fitness)
        return global_best
