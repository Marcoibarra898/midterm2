import unittest
from src.geometry import Point, euclidean_distance, calculate_tour_distance
from src.spatial import KDTree
from src.tsp_solver import GeneticAlgorithm, two_opt, nearest_neighbor_init

class TestGeometry(unittest.TestCase):
    def test_distance(self):
        p1 = Point(0, 0, 0)
        p2 = Point(1, 3, 4)
        self.assertEqual(p1.distance_to(p2), 5.0)

class TestKDTree(unittest.TestCase):
    def test_nearest_neighbor(self):
        points = [
            Point(0, 0, 0),
            Point(1, 10, 10),
            Point(2, 20, 20),
            Point(3, 5, 5)
        ]
        tree = KDTree(points)
        target = Point(99, 1, 1)
        nearest = tree.nearest_neighbor(target)
        self.assertEqual(nearest.id, 0) # (0,0) is closest to (1,1)
        
        target2 = Point(99, 6, 6)
        nearest2 = tree.nearest_neighbor(target2)
        self.assertEqual(nearest2.id, 3) # (5,5) is closest to (6,6)

class TestTSPSolver(unittest.TestCase):
    def setUp(self):
        self.points = [
            Point(0, 0, 0),
            Point(1, 0, 1),
            Point(2, 1, 1),
            Point(3, 1, 0)
        ] # Square
        
    def test_tour_distance(self):
        # 0->1->2->3->0 = 1+1+1+1 = 4
        dist = calculate_tour_distance(self.points)
        self.assertEqual(dist, 4.0)

    def test_two_opt_optimization(self):
        # Crossed tour: 0->2->1->3->0 (Diagonals: sqrt(2)*2 + 1 + 1 = 2.82 + 2 = 4.82)
        crossed_tour = [self.points[0], self.points[2], self.points[1], self.points[3]]
        optimized = two_opt(crossed_tour)
        dist = calculate_tour_distance(optimized)
        self.assertAlmostEqual(dist, 4.0) # Should untangle to square

    def test_ga_initialization(self):
        ga = GeneticAlgorithm(self.points, pop_size=10)
        self.assertEqual(len(ga.population), 10)
        self.assertEqual(len(ga.population[0].tour), 4)

if __name__ == '__main__':
    unittest.main()
