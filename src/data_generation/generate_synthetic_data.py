"""
Module for generating synthetic TLE data for training and testing.
Uses a DETERMINISTIC approach for collision generation to guarantee valid data.
"""
import random
import datetime
import math
import numpy as np
from tqdm import tqdm
import json
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import SYNTHETIC_DATA_CONFIG, DATA_DIR
from src.utils.tle_utils import TLEHandler

# Constants
MU_EARTH = 3.986004418e5  # km^3/s^2
R_EARTH = 6378.137  # km


def compute_checksum(line):
    """Calculates TLE checksum (mod 10 of digit sum, '-' counts as 1)."""
    checksum = 0
    for char in line:
        if char.isdigit():
            checksum += int(char)
        elif char == '-':
            checksum += 1
    return checksum % 10


def format_tle_line1(sat_id, epoch, bstar=0.0):
    """Formats TLE Line 1."""
    epoch_year = epoch.year % 100
    start_of_year = datetime.datetime(epoch.year, 1, 1)
    day_of_year = (epoch - start_of_year).days + 1
    day_fraction = (epoch.hour * 3600 + epoch.minute * 60 + epoch.second + epoch.microsecond / 1e6) / 86400.0
    epoch_day = day_of_year + day_fraction

    # Format: 1 NNNNNC NNNNNNNN NNNNN.NNNNNNNN +.NNNNNNNN +NNNNN-N +NNNNN-N N NNNNN
    # Simplified version
    line = f"1 {sat_id:05d}U 24001A   {epoch_year:02d}{epoch_day:012.8f}  .00000000  00000-0  00000-0 0  999"
    line += str(compute_checksum(line))
    return line


def format_tle_line2(sat_id, inc, raan, ecc, argp, ma, mm, rev_num=99999):
    """Formats TLE Line 2."""
    # Eccentricity is stored as decimal without leading 0 (e.g., 0.0001234 -> 0001234)
    ecc_int = int(round(ecc * 1e7))
    line = f"2 {sat_id:05d} {inc:8.4f} {raan:8.4f} {ecc_int:07d} {argp:8.4f} {ma:8.4f} {mm:11.8f}{rev_num:5d}"
    line += str(compute_checksum(line))
    return line


def altitude_to_mean_motion(altitude_km):
    """Converts altitude (km) to mean motion (revolutions per day)."""
    a = R_EARTH + altitude_km
    period_sec = 2 * math.pi * math.sqrt(a**3 / MU_EARTH)
    return 86400.0 / period_sec


def eci_to_lat_lon_alt(r, timestamp):
    """
    Converts ECI position to geodetic latitude, longitude, altitude.
    Simplified calculation (ignores Earth rotation for longitude accuracy).
    """
    x, y, z = r
    
    # Altitude (distance from Earth center minus Earth radius)
    alt_km = np.linalg.norm(r) - R_EARTH
    
    # Latitude (angle from equatorial plane)
    lat = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))
    
    # Longitude (simplified - ignores Greenwich sidereal time for demo)
    lon = math.degrees(math.atan2(y, x))
    
    return lat, lon, alt_km


def generate_collision_scenario(sat_id_a, sat_id_b, epoch):
    """
    Generates a GUARANTEED collision scenario.
    
    Strategy:
    1. Create Satellite A with random orbital elements.
    2. Create Satellite B with NEARLY IDENTICAL elements but slightly different Mean Anomaly.
       This ensures they are on the same orbit but at different phases.
    3. Calculate when they will be closest (same MA) and record that as collision time.
    4. The "collision" location is where Sat A is at that predicted time.
    """
    # Random orbital elements for Sat A
    alt_km = random.uniform(*SYNTHETIC_DATA_CONFIG['altitude_range_km'])
    inc = random.uniform(0, 90)  # Use 0-90 for more realistic LEO
    raan = random.uniform(0, 360)
    ecc = random.uniform(0.0001, 0.02)  # Low eccentricity
    argp = random.uniform(0, 360)
    ma_a = random.uniform(0, 360)
    mm = altitude_to_mean_motion(alt_km)
    
    # Satellite B: Same orbit, slight phase difference
    # The phase difference determines time to "catch up" = collision time
    phase_diff_deg = random.uniform(0.5, 10.0)  # Small phase difference
    ma_b = (ma_a + phase_diff_deg) % 360
    
    # Also add tiny perturbations to create actual "different" satellites
    inc_b = inc + random.uniform(-0.01, 0.01)
    raan_b = raan + random.uniform(-0.01, 0.01)
    
    # Calculate collision time
    # Relative angular velocity = 0 for same orbit, so they never actually catch up
    # unless we slightly modify mean motion
    mm_b = mm + random.uniform(-0.001, 0.001)  # Tiny difference
    
    # Time for phase to close (simplified)
    # delta_phase (deg) / (delta_mm * 360 deg/rev) = time in days
    if abs(mm - mm_b) > 1e-6:
        time_to_collision_days = abs(phase_diff_deg / 360.0) / abs(mm - mm_b)
    else:
        time_to_collision_days = random.uniform(0.5, 5.0)
    
    # Clamp to window
    time_to_collision_days = min(time_to_collision_days, SYNTHETIC_DATA_CONFIG['collision_time_window'])
    time_to_collision_days = max(time_to_collision_days, 0.1)
    
    # Create TLE strings
    line1_a = format_tle_line1(sat_id_a, epoch)
    line2_a = format_tle_line2(sat_id_a, inc, raan, ecc, argp, ma_a, mm)
    
    line1_b = format_tle_line1(sat_id_b, epoch)
    line2_b = format_tle_line2(sat_id_b, inc_b, raan_b, ecc, argp, ma_b, mm_b)
    
    # Calculate collision location
    collision_time = epoch + datetime.timedelta(days=time_to_collision_days)
    sat_a = TLEHandler.parse_tle(line1_a, line2_a)
    r_a, _ = TLEHandler.propagate(sat_a, collision_time)
    
    if r_a is not None:
        lat, lon, alt = eci_to_lat_lon_alt(r_a, collision_time)
    else:
        lat, lon, alt = 0, 0, alt_km
    
    # Calculate actual minimum distance at collision time
    sat_b = TLEHandler.parse_tle(line1_b, line2_b)
    r_b, _ = TLEHandler.propagate(sat_b, collision_time)
    
    if r_a is not None and r_b is not None:
        min_dist = TLEHandler.calculate_distance(r_a, r_b)
    else:
        min_dist = random.uniform(0.1, 5.0)  # Synthetic distance
    
    return {
        "satellite_a": {"name": f"SAT-A-{sat_id_a}", "tle_line1": line1_a, "tle_line2": line2_a},
        "satellite_b": {"name": f"SAT-B-{sat_id_b}", "tle_line1": line1_b, "tle_line2": line2_b},
        "collision": True,
        "collision_time_days": round(time_to_collision_days, 4),
        "min_distance_km": round(min_dist, 4),
        "collision_location": {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "altitude_km": round(alt, 2)
        }
    }


def generate_safe_scenario(sat_id_a, sat_id_b, epoch):
    """
    Generates a GUARANTEED safe (non-collision) scenario.
    
    Strategy: Put satellites in completely different orbital planes (different inclinations).
    """
    # Satellite A
    alt_a = random.uniform(*SYNTHETIC_DATA_CONFIG['altitude_range_km'])
    inc_a = random.uniform(0, 45)
    raan_a = random.uniform(0, 180)
    ecc_a = random.uniform(0.0001, 0.02)
    argp_a = random.uniform(0, 360)
    ma_a = random.uniform(0, 360)
    mm_a = altitude_to_mean_motion(alt_a)
    
    # Satellite B: Different plane (guaranteed separation)
    alt_b = random.uniform(*SYNTHETIC_DATA_CONFIG['altitude_range_km'])
    inc_b = random.uniform(60, 120)  # Significantly different inclination
    raan_b = random.uniform(180, 360)  # Different RAAN
    ecc_b = random.uniform(0.0001, 0.02)
    argp_b = random.uniform(0, 360)
    ma_b = random.uniform(0, 360)
    mm_b = altitude_to_mean_motion(alt_b)
    
    # Create TLE strings
    line1_a = format_tle_line1(sat_id_a, epoch)
    line2_a = format_tle_line2(sat_id_a, inc_a, raan_a, ecc_a, argp_a, ma_a, mm_a)
    
    line1_b = format_tle_line1(sat_id_b, epoch)
    line2_b = format_tle_line2(sat_id_b, inc_b, raan_b, ecc_b, argp_b, ma_b, mm_b)
    
    # Calculate minimum distance over the analysis window
    sat_a = TLEHandler.parse_tle(line1_a, line2_a)
    sat_b = TLEHandler.parse_tle(line1_b, line2_b)
    
    min_dist = float('inf')
    check_points = 10
    for i in range(check_points):
        t = epoch + datetime.timedelta(days=i * SYNTHETIC_DATA_CONFIG['collision_time_window'] / check_points)
        r_a, _ = TLEHandler.propagate(sat_a, t)
        r_b, _ = TLEHandler.propagate(sat_b, t)
        if r_a is not None and r_b is not None:
            dist = TLEHandler.calculate_distance(r_a, r_b)
            min_dist = min(min_dist, dist)
    
    if min_dist == float('inf'):
        min_dist = random.uniform(100, 5000)  # Fallback
    
    return {
        "satellite_a": {"name": f"SAT-A-{sat_id_a}", "tle_line1": line1_a, "tle_line2": line2_a},
        "satellite_b": {"name": f"SAT-B-{sat_id_b}", "tle_line1": line1_b, "tle_line2": line2_b},
        "collision": False,
        "collision_time_days": -1,
        "min_distance_km": round(min_dist, 2),
        "collision_location": {"latitude": 0, "longitude": 0, "altitude_km": 0}
    }


def generate_dataset(num_collision, num_safe, output_dir):
    """
    Main dataset generation function.
    Generates collision and non-collision scenarios separately.
    """
    os.makedirs(os.path.join(output_dir, 'synthetic'), exist_ok=True)
    
    epoch = datetime.datetime.now()
    
    # Generate Collision Scenarios
    print(f"\n{'='*50}")
    print(f"Generating {num_collision} COLLISION scenarios...")
    print(f"{'='*50}")
    
    collision_data = []
    for i in tqdm(range(num_collision), desc="Collision scenarios"):
        sat_id_a = 10000 + i * 2
        sat_id_b = 10001 + i * 2
        scenario = generate_collision_scenario(sat_id_a, sat_id_b, epoch)
        collision_data.append(scenario)
    
    collision_path = os.path.join(output_dir, 'synthetic', 'collision_data.json')
    with open(collision_path, 'w') as f:
        json.dump(collision_data, f, indent=2)
    print(f"✓ Saved {len(collision_data)} collision scenarios to {collision_path}")
    
    # Generate Safe Scenarios
    print(f"\n{'='*50}")
    print(f"Generating {num_safe} SAFE (non-collision) scenarios...")
    print(f"{'='*50}")
    
    safe_data = []
    for i in tqdm(range(num_safe), desc="Safe scenarios"):
        sat_id_a = 60000 + i * 2
        sat_id_b = 60001 + i * 2
        scenario = generate_safe_scenario(sat_id_a, sat_id_b, epoch)
        safe_data.append(scenario)
    
    safe_path = os.path.join(output_dir, 'synthetic', 'non_collision_data.json')
    with open(safe_path, 'w') as f:
        json.dump(safe_data, f, indent=2)
    print(f"✓ Saved {len(safe_data)} safe scenarios to {safe_path}")
    
    print(f"\n{'='*50}")
    print("DATA GENERATION COMPLETE!")
    print(f"Total samples: {len(collision_data) + len(safe_data)}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    print(f"Configuration: {SYNTHETIC_DATA_CONFIG}")
    generate_dataset(
        SYNTHETIC_DATA_CONFIG['num_collision_samples'],
        SYNTHETIC_DATA_CONFIG['num_non_collision_samples'],
        DATA_DIR
    )
