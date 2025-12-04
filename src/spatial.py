from typing import List, Optional, Tuple
from geometry import Point
import math

class KDNode:
    def __init__(self, point: Point, left: Optional['KDNode'] = None, right: Optional['KDNode'] = None, axis: int = 0):
        self.point = point
        self.left = left
        self.right = right
        self.axis = axis

class KDTree:
    """
    K-Dimensional Tree for efficient spatial queries (Nearest Neighbor).
    Implemented specifically for 2D points (k=2).
    """
    def __init__(self, points: List[Point]):
        self.k = 2
        self.root = self._build(points, 0)

    def _build(self, points: List[Point], depth: int) -> Optional[KDNode]:
        """Recursively builds the K-D Tree."""
        if not points:
            return None

        axis = depth % self.k
        # Sort point list and choose median as pivot element
        points.sort(key=lambda p: p.x if axis == 0 else p.y)
        median = len(points) // 2

        return KDNode(
            point=points[median],
            left=self._build(points[:median], depth + 1),
            right=self._build(points[median + 1:], depth + 1),
            axis=axis
        )

    def nearest_neighbor(self, target: Point, exclude_set: set = None) -> Optional[Point]:
        """Finds the nearest neighbor to target that is NOT in exclude_set."""
        best_node = None
        best_dist = float('inf')

        def recursive_search(node: Optional[KDNode]):
            nonlocal best_node, best_dist
            if node is None:
                return

            # Calculate distance
            dist = target.distance_to(node.point)
            
            # Update best if closer and not excluded
            if dist < best_dist and (exclude_set is None or node.point.id not in exclude_set):
                best_dist = dist
                best_node = node.point

            # Determine which subtree to search first
            axis = node.axis
            target_coord = target.x if axis == 0 else target.y
            node_coord = node.point.x if axis == 0 else node.point.y
            
            diff = target_coord - node_coord

            close_branch = node.left if diff < 0 else node.right
            far_branch = node.right if diff < 0 else node.left

            recursive_search(close_branch)

            # Only check the far branch if there's a possibility of a closer point
            if abs(diff) < best_dist:
                recursive_search(far_branch)

        recursive_search(self.root)
        return best_node
