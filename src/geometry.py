import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Point:
    """Represents a city in 2D space."""
    id: int
    x: float
    y: float

    def distance_to(self, other: 'Point') -> float:
        """Calculates Euclidean distance to another point."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def to_array(self) -> np.ndarray:
        """Returns coordinates as a numpy array."""
        return np.array([self.x, self.y])

def euclidean_distance(p1: Point, p2: Point) -> float:
    """Computes Euclidean distance between two points."""
    return p1.distance_to(p2)

def generate_random_cities(n: int, width: float = 1000.0, height: float = 1000.0, seed: int = None) -> List[Point]:
    """Generates n random cities within a given area."""
    if seed is not None:
        np.random.seed(seed)
    
    cities = []
    for i in range(n):
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        cities.append(Point(i, x, y))
    return cities

def calculate_tour_distance(tour: List[Point]) -> float:
    """Calculates the total distance of a tour (round trip)."""
    total_dist = 0.0
    for i in range(len(tour)):
        total_dist += tour[i].distance_to(tour[(i + 1) % len(tour)])
    return total_dist
