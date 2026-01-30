"""
Feature extraction module for converting TLE pairs into ML-ready feature vectors.
"""
import numpy as np
import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.tle_utils import TLEHandler
from config import FEATURE_CONFIG

class FeatureExtractor:
    @staticmethod
    def extract_features(tle_a_line1, tle_a_line2, tle_b_line1, tle_b_line2):
        """
        Extracts features from two TLEs.
        
        Returns:
            np.array: Feature vector of shape (42,)
            dict: Meta info (min_dist, time_to_closest)
        """
        # Parse TLEs
        sat_a = TLEHandler.parse_tle(tle_a_line1, tle_a_line2)
        sat_b = TLEHandler.parse_tle(tle_b_line1, tle_b_line2)
        
        # 1. Orbital Elements Features (12 features)
        oe_a = TLEHandler.get_orbital_elements(sat_a)
        oe_b = TLEHandler.get_orbital_elements(sat_b)
        
        features = []
        
        # Add raw orbital elements for both
        for oe in [oe_a, oe_b]:
            features.extend([
                oe['inclination'], oe['raan'], oe['eccentricity'], 
                oe['arg_perigee'], oe['mean_anomaly'], oe['mean_motion']
            ])
            
        # 2. Relative Orbital Elements (6 features)
        features.extend([
            oe_a['inclination'] - oe_b['inclination'],
            oe_a['raan'] - oe_b['raan'],
            oe_a['eccentricity'] - oe_b['eccentricity'],
            oe_a['arg_perigee'] - oe_b['arg_perigee'],
            oe_a['mean_anomaly'] - oe_b['mean_anomaly'],
            oe_a['mean_motion'] - oe_b['mean_motion']
        ])
        
        # 3. Propagated State Features (24 features)
        # We propagate to now and a few steps ahead to capture relative motion dynamics
        
        now = datetime.datetime.now()
        
        # Get state at t=0 (now)
        r_a, v_a = TLEHandler.propagate(sat_a, now)
        r_b, v_b = TLEHandler.propagate(sat_b, now)
        
        if r_a is None or r_b is None:
            # Handle propagation errors - return zeros or raise
            return np.zeros(42), {}
            
        # Relative position and velocity at t=0
        r_rel = r_a - r_b
        v_rel = v_a - v_b
        dist = np.linalg.norm(r_rel)
        speed_rel = np.linalg.norm(v_rel)
        
        features.extend(r_rel.tolist()) # 3
        features.extend(v_rel.tolist()) # 3
        features.append(dist)           # 1
        features.append(speed_rel)      # 1
        
        # Propagate to future points to capture convergence/divergence
        # e.g., +1 day, +3 days
        for days in [1, 3]:
            t_fut = now + datetime.timedelta(days=days)
            r_a_fut, v_a_fut = TLEHandler.propagate(sat_a, t_fut)
            r_b_fut, v_b_fut = TLEHandler.propagate(sat_b, t_fut)
            
            if r_a_fut is None or r_b_fut is None:
                features.extend([0]*8)
                continue
                
            r_rel_fut = r_a_fut - r_b_fut
            v_rel_fut = v_a_fut - v_b_fut
            dist_fut = np.linalg.norm(r_rel_fut)
            speed_rel_fut = np.linalg.norm(v_rel_fut)
            
            features.extend(r_rel_fut.tolist()) # 3
            features.extend(v_rel_fut.tolist()) # 3
            features.append(dist_fut)           # 1
            features.append(speed_rel_fut)      # 1

        # Total features so far: 12 + 6 + 8 + 8 + 8 = 42?
        # 12 (OE) + 6 (Rel OE) + 8 (t0) + 8 (t1) + 8 (t2) = 42. Exactly.
        
        return np.array(features), {'current_dist': dist}

if __name__ == "__main__":
    # Test
    # Create dummy TLEs (ISS-like)
    l1 = "1 25544U 98067A   23001.00000000  .00016717  00000-0  10270-3 0  9999"
    l2 = "2 25544  51.6400 208.9163 0006703  69.9862  25.2906 15.54225995123456"
    
    feats, _ = FeatureExtractor.extract_features(l1, l2, l1, l2) # Self vs Self
    print(f"Features shape: {feats.shape}")
    print(f"First 10 features: {feats[:10]}")
