import numpy as np
import random
from typing import List, Set
from src.geometry import Point, calculate_tour_distance

class QLearningTSPSolver:
    def __init__(self, points: List[Point], alpha=0.1, gamma=0.99, epsilon=0.1):
        self.points = points
        self.num_cities = len(points)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Q-Table: [Current City][Next City]
        # Initialize with zeros
        self.q_table = np.zeros((self.num_cities, self.num_cities))
        
        # Precompute distances for efficiency
        self.dist_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    self.dist_matrix[i][j] = points[i].distance_to(points[j])
                else:
                    self.dist_matrix[i][j] = float('inf') # Avoid self-loops

    def train(self, episodes: int = 1000):
        """
        Trains the Q-Table for a specified number of episodes.
        """
        for episode in range(episodes):
            # Start at a random city
            start_node = random.randint(0, self.num_cities - 1)
            current_node = start_node
            visited = {current_node}
            path = [current_node]
            
            # Decay epsilon? Optional, but good for convergence
            # self.epsilon = max(0.01, self.epsilon * 0.9995)

            while len(visited) < self.num_cities:
                # Choose action
                next_node = self._choose_action(current_node, visited)
                
                # Calculate reward (negative distance)
                # We want to minimize distance, so reward is -distance
                dist = self.dist_matrix[current_node][next_node]
                reward = -dist
                
                # Update Q-Value
                # State: current_node, Action: next_node
                # Next State: next_node
                # Possible actions from next_node: all unvisited nodes (excluding next_node itself)
                # But wait, Q-Learning usually considers max Q over all possible actions from next state.
                # In TSP, the set of possible actions changes (unvisited set shrinks).
                # Standard Q-Learning doesn't handle changing action spaces naturally without state including visited set.
                # However, for simple TSP approximation, we can just look at max Q of *currently* valid moves?
                # Or just max Q over all columns (ignoring visited constraint for update)? 
                # Ignoring visited constraint in update is a common simplification for tabular TSP, 
                # but let's try to be slightly more accurate: max Q over all nodes except next_node (since we can't go back immediately usually, but actually we can go anywhere unvisited).
                # Let's use max over all columns for simplicity, as the "potential" of the city.
                
                max_next_q = np.max(self.q_table[next_node])
                
                current_q = self.q_table[current_node][next_node]
                new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
                self.q_table[current_node][next_node] = new_q
                
                # Move to next state
                visited.add(next_node)
                path.append(next_node)
                current_node = next_node
            
            # Complete the tour (return to start)
            dist = self.dist_matrix[current_node][start_node]
            reward = -dist
            # No next state, terminal
            current_q = self.q_table[current_node][start_node]
            new_q = current_q + self.alpha * (reward - current_q) # gamma * 0
            self.q_table[current_node][start_node] = new_q

    def _choose_action(self, current_node: int, visited: Set[int]) -> int:
        """
        Epsilon-greedy action selection, masked by unvisited cities.
        """
        unvisited = [i for i in range(self.num_cities) if i not in visited]
        
        if not unvisited:
            return -1 # Should not happen in loop
            
        if random.random() < self.epsilon:
            return random.choice(unvisited)
        else:
            # Greedy: Choose unvisited with max Q value
            # Filter Q-values for unvisited
            q_values = self.q_table[current_node, unvisited]
            # Find index of max in q_values, then map back to original index
            max_idx = np.argmax(q_values)
            return unvisited[max_idx]

    def get_solution(self) -> List[Point]:
        """
        Generates the best tour based on the learned Q-Table.
        Starts at city 0 (or random) and follows max Q greedy policy.
        """
        start_node = 0
        current_node = start_node
        visited = {current_node}
        tour_indices = [current_node]
        
        while len(visited) < self.num_cities:
            unvisited = [i for i in range(self.num_cities) if i not in visited]
            
            # Pure greedy based on Q-table
            q_values = self.q_table[current_node, unvisited]
            max_idx = np.argmax(q_values)
            next_node = unvisited[max_idx]
            
            visited.add(next_node)
            tour_indices.append(next_node)
            current_node = next_node
            
        # Convert indices back to Point objects
        tour = [self.points[i] for i in tour_indices]
        return tour
