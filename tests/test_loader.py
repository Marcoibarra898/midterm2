import unittest
import os
from src.geometry import load_cities_from_csv

class TestLoader(unittest.TestCase):
    def test_load_csv(self):
        # Create a dummy CSV file
        filename = "test_data.csv"
        with open(filename, "w") as f:
            f.write("PULocationID,dispatching_base_num\n")
            f.write("1,B00001\n")
            f.write("2,B00002\n")
            f.write("1,B00003\n") # Duplicate ID
            
        try:
            cities = load_cities_from_csv(filename)
            self.assertEqual(len(cities), 2) # Should be 2 unique IDs (1 and 2)
            ids = sorted([c.id for c in cities])
            self.assertEqual(ids, [1, 2])
        finally:
            if os.path.exists(filename):
                os.remove(filename)

if __name__ == '__main__':
    unittest.main()
