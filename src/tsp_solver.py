import random
import numpy as np
from typing import List, Tuple
from geometry import Point, calculate_tour_distance
from spatial import KDTree

class Individual:
    def __init__(self, tour: List[Point]):
        self.tour = tour
        self.fitness = calculate_tour_distance(tour)

    def __lt__(self, other):
        return self.fitness < other.fitness

def nearest_neighbor_init(points: List[Point], kdtree: KDTree) -> List[Point]:
    """
    [Greedy Algorithm] Generates a tour using the Nearest Neighbor heuristic.
    Uses K-D Tree for efficient lookups.
    """
    if not points:
        return []
    
    unvisited = set(p.id for p in points)
    current = random.choice(points)
    tour = [current]
    unvisited.remove(current.id)

    while unvisited:
        # Find nearest neighbor that is in unvisited set
        # We query k=len(points) effectively, but KDTree optimization helps
        # In practice for NN, we just need the closest valid one.
        # Since our KDTree returns single NN, we might need to query iteratively or just scan if KDTree doesn't support set exclusion efficiently enough.
        # Our KDTree supports exclude_set!
        
        next_city = kdtree.nearest_neighbor(current, exclude_set=None) # Optimization: Pass visited set as exclusion?
        # Actually our KDTree.nearest_neighbor takes exclude_set as 'visited' implicitly if we invert logic
        # But for simplicity and performance in this specific KDTree implementation:
        # Let's rely on the fact that we need to find the nearest *unvisited*.
        # If KDTree doesn't support efficient unvisited filtering, we might fallback to linear scan for very small sets, 
        # but for large sets we want the tree.
        # Let's use the exclude_set feature we added to KDTree.
        
        # We need to pass the set of *visited* IDs to exclude.
        visited_ids = set(p.id for p in tour)
        next_city = kdtree.nearest_neighbor(current, exclude_set=visited_ids)
        
        if next_city:
            tour.append(next_city)
            unvisited.remove(next_city.id)
            current = next_city
        else:
            # Should not happen unless graph is disconnected or logic error, 
            # but if it does, pick random unvisited
            if unvisited:
                next_id = unvisited.pop()
                # Find point object
                next_city = next(p for p in points if p.id == next_id)
                tour.append(next_city)
                current = next_city
    
    return tour

def two_opt(tour: List[Point]) -> List[Point]:
    """
    [Local Search] Applies 2-Opt optimization to remove crossing edges.
    """
    best_tour = tour[:]
    best_dist = calculate_tour_distance(best_tour)
    improved = True
    
    while improved:
        improved = False
        for i in range(1, len(best_tour) - 1):
            for j in range(i + 1, len(best_tour)):
                # Check if swap improves distance
                # Dist(A, B) + Dist(C, D) > Dist(A, C) + Dist(B, D)
                # A=i-1, B=i, C=j, D=j+1
                pA = best_tour[i-1]
                pB = best_tour[i]
                pC = best_tour[j]
                pD = best_tour[(j+1) % len(best_tour)]
                
                d1 = pA.distance_to(pB) + pC.distance_to(pD)
                d2 = pA.distance_to(pC) + pB.distance_to(pD)
                
                if d2 < d1:
                    # Perform swap: reverse segment [i, j]
                    best_tour[i:j+1] = reversed(best_tour[i:j+1])
                    best_dist -= (d1 - d2)
                    improved = True
    
    return best_tour

class GeneticAlgorithm:
    def __init__(self, points: List[Point], pop_size: int = 50, elite_size: int = 5, mutation_rate: float = 0.1):
        self.points = points
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.kdtree = KDTree(points)
        self.population = self._init_population()

    def _init_population(self) -> List[Individual]:
        pop = []
        # 1. Greedy Initialization (10% of population)
        num_greedy = max(1, int(0.1 * self.pop_size))
        for _ in range(num_greedy):
            tour = nearest_neighbor_init(self.points, self.kdtree)
            pop.append(Individual(tour))
        
        # 2. Random Initialization (Rest)
        for _ in range(self.pop_size - num_greedy):
            tour = self.points[:]
            random.shuffle(tour)
            pop.append(Individual(tour))
        
        return pop

    def evolve(self) -> Individual:
        # Selection (Tournament)
        new_pop = []
        
        # Elitism
        self.population.sort()
        new_pop.extend(self.population[:self.elite_size])
        
        while len(new_pop) < self.pop_size:
            p1 = self._tournament_select()
            p2 = self._tournament_select()
            
            # Crossover (OX1)
            child_tour = self._crossover_ox1(p1.tour, p2.tour)
            
            # Mutation
            if random.random() < self.mutation_rate:
                child_tour = self._mutate(child_tour)
            
            new_pop.append(Individual(child_tour))
            
        self.population = new_pop
        return self.population[0] # Return best

    def _tournament_select(self, k=3) -> Individual:
        candidates = random.sample(self.population, k)
        return min(candidates, key=lambda ind: ind.fitness)

    def _crossover_ox1(self, p1: List[Point], p2: List[Point]) -> List[Point]:
        size = len(p1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [None] * size
        child[start:end] = p1[start:end]
        
        current_p2_idx = 0
        for i in range(size):
            if child[i] is None:
                while p2[current_p2_idx] in child[start:end]: # Note: This check is O(N), optimized sets would be better for large N
                    current_p2_idx += 1
                child[i] = p2[current_p2_idx]
                current_p2_idx += 1
        return child

    def _mutate(self, tour: List[Point]) -> List[Point]:
        # Swap Mutation
        idx1, idx2 = random.sample(range(len(tour)), 2)
        tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
        return tour

    def run_local_search(self):
        """Applies 2-Opt to the best individual."""
        best = self.population[0]
        optimized_tour = two_opt(best.tour)
        self.population[0] = Individual(optimized_tour)
