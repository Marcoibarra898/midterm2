import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.geometry import generate_random_cities, calculate_tour_distance
from src.rl_solver import QLearningTSPSolver
import time

def test_rl_solver():
    print("Generating 10 random cities...")
    points = generate_random_cities(10, seed=42)
    
    print("Initializing Q-Learning Solver...")
    solver = QLearningTSPSolver(points, alpha=0.1, gamma=0.9, epsilon=0.1)
    
    print("Training for 1000 episodes...")
    start_time = time.time()
    solver.train(episodes=1000)
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.4f} seconds.")
    
    print("Generating solution...")
    tour = solver.get_solution()
    
    assert len(tour) == 10, "Tour should contain all 10 cities"
    assert len(set(p.id for p in tour)) == 10, "Tour should contain unique cities"
    
    dist = calculate_tour_distance(tour)
    print(f"Tour Distance: {dist:.2f}")
    print("Test Passed!")

if __name__ == "__main__":
    test_rl_solver()
