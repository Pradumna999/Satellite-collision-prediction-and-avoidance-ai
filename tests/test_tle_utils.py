import unittest
from datetime import datetime
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.tle_utils import TLEHandler

class TestTLEUtils(unittest.TestCase):
    def setUp(self):
        # ISS TLE (Example)
        self.line1 = "1 25544U 98067A   23001.00000000  .00016717  00000-0  10270-3 0  9999"
        # Checksum might need correction if string is modified manually, SGP4 parser is lenient usually
        self.line2 = "2 25544  51.6400 208.9163 0006703  69.9862  25.2906 15.54225995123456"
        
    def test_parse_and_propagate(self):
        sat = TLEHandler.parse_tle(self.line1, self.line2)
        self.assertIsNotNone(sat)
        
        # Propagate
        dt = datetime(2023, 1, 1, 0, 0, 0)
        r, v = TLEHandler.propagate(sat, dt)
        
        self.assertIsNotNone(r)
        self.assertIsNotNone(v)
        self.assertEqual(len(r), 3)
        self.assertEqual(len(v), 3)
        
        # Check altitude is reasonable (~400km + 6378km earth radius = ~6778 km from center)
        dist = np.linalg.norm(r)
        self.assertTrue(6600 < dist < 7000, f"Distance {dist} km seems off for LEO")

if __name__ == '__main__':
    unittest.main()
