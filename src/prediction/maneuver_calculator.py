"""
Module for calculating collision avoidance maneuvers.
"""
import math
import numpy as np
import datetime
from src.utils.tle_utils import TLEHandler

class ManeuverCalculator:
    def __init__(self, config):
        """
        Args:
            config (dict): MANEUVER_CONFIG from config.py
        """
        self.config = config
        
    def calculate_maneuver(self, satellite, collision_time, miss_distance):
        """
        Calculates recommended maneuver.
        
        Types:
        1. Altitude Change (Hohmann-like transfer to change period/phasing)
           - Applied 1/2 orbit before collision or much earlier?
           - Usually "Phase Change" = slight semi-major axis change to change period, drift, then restore?
           - Simplified: Change altitude by X km to avoid collision.
           
        Args:
            satellite (Satrec): The satellite to maneuver
            collision_time (datetime): Predicted collision time
            miss_distance (float): Predicted miss distance (km) [Currently unused vs threshold]
            
        Returns:
            dict: Maneuver details
        """
        # Assumptions for simplified calculation:
        # We want to change position by > 10 km (safety margin) at collision time.
        # Most efficient: Change velocity along track (prograde/retrograde) well in advance.
        
        # Lead time
        lead_time_hours = self.config['lead_time_hours']
        maneuver_time = collision_time - datetime.timedelta(hours=lead_time_hours)
        
        if maneuver_time < datetime.datetime.now():
            # Emergency: Immediate maneuver
            maneuver_time = datetime.datetime.now()
            hours_avail = (collision_time - maneuver_time).total_seconds() / 3600.0
            type = "emergency_impulse"
        else:
            type = "altitude_change"
            hours_avail = lead_time_hours
            
        # Physics approximation:
        # Along-track shift (Delta L) ~= -3 * Pi * (Delta a / a) * N_revs
        # We need Delta L ~ 10 km (+ radius of uncertainty)
        # Delta a = (Delta L * a) / (-3 * Pi * N_revs)
        
        # Get semi-major axis (a)
        # n (revs/day)
        n = satellite.no_kozai * 1440.0 / (2 * np.pi) # convert rad/min to rev/day
        if n == 0: n = 15.5 # Fallback
        
        # a = (mu / n^2)^(1/3)
        mu = 3.986004418e5
        # n in rad/s
        n_rad_s = satellite.no_kozai / 60.0
        a = (mu / n_rad_s**2)**(1/3)
        
        target_sep = self.config['safety_margin_km']
        
        # Number of revs until collision
        n_revs = (hours_avail / 24.0) * n
        
        if n_revs < 0.1:
            # Too late for phase change, need radial/normal thrust (expensive)
            req_delta_v = 10.0 # m/s (placeholder for high impulse)
            direction = "normal"
        else:
            # Phase change required
            # Delta a approx
            # Delta L = target_sep
            # |Delta a| = (target_sep * a) / (3 * pi * n_revs)
            
            delta_a = (target_sep * a) / (3 * np.pi * n_revs)
            
            # Delta V required to change 'a' by 'delta_a'
            # Vis-viva: v = sqrt(mu/a)
            # dv/da = -1/2 * sqrt(mu) * a^(-3/2) = -v / (2a)
            # dv = (v / 2a) * da
            
            v = np.sqrt(mu / a) # km/s
            delta_v_kms = (v / (2*a)) * delta_a
            
            # Minimum check
            if delta_v_kms * 1000 < self.config['min_delta_v']:
                delta_v_kms = self.config['min_delta_v'] / 1000.0
                
            req_delta_v = delta_v_kms * 1000.0 # m/s
            direction = "prograde"
            
        success_prob = 100.0 * (1 - math.exp(-req_delta_v)) # Dummy prob curve
        if success_prob > 99.9: success_prob = 99.9
        
        return {
            "type": type,
            "direction": direction,
            "delta_v": round(req_delta_v, 2),
            "execute_time": maneuver_time.strftime("%Y-%m-%d %H:%M:%S"),
            "success_probability": round(success_prob, 1),
            "precautions": [
                "Verify post-maneuver trajectory",
                "Monitor conjunction for 72 hours",
                "Notify space traffic management"
            ]
        }
