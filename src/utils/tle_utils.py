"""
Utilities for TLE parsing and SGP4 propagation.
"""
import numpy as np
from sgp4.api import Satrec, WGS72
from sgp4.exporter import export_tle
from datetime import datetime, timedelta

class TLEHandler:
    @staticmethod
    def parse_tle(line1, line2):
        """Parses TLE lines into a Satrec object."""
        satellite = Satrec.twoline2rv(line1, line2)
        return satellite

    @staticmethod
    def propagate(satellite, timestamp):
        """
        Propagates satellite to a specific datetime.
        Returns position (km) and velocity (km/s) vectors in ECI (TEME) frame.
        """
        # SGP4 expects julian date parameters
        jd, fr = TLEHandler.datetime_to_jd(timestamp)
        e, r, v = satellite.sgp4(jd, fr)
        
        if e != 0:
            # Error in propagation
            return None, None
            
        return np.array(r), np.array(v)

    @staticmethod
    def datetime_to_jd(dt):
        """Converts datetime to Julian Date parts (jd, fr)."""
        # SGP4 python library documentation recommends this way if using jplephem or similar, 
        # but for pure sgp4 2.x, we can use built-in helper or standard formula.
        # Using a standard conversion here for dependency minimization.
        # Julian date of 2000-01-01 12:00:00 is 2451545.0
        
        # Simplified conversion for modern python (using sgp4's internal if avail or manual)
        # Using the method from sgp4 library docs would be best if `sgp4.api` exposed it directly,
        # but `jday` is in `sgp4.api`.
        from sgp4.api import jday
        
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)
        return jd, fr

    @staticmethod
    def calculate_distance(r1, r2):
        """Calculates Euclidean distance between two position vectors."""
        return np.linalg.norm(r1 - r2)

    @staticmethod
    def create_dummy_tle(catalog_number, epoch_date):
        """
        Creates a dummy valid Satrec object for testing/generation.
        This provides a base to modify orbital elements directly if needed,
        BUT SGP4 Satrec objects are optimized C++ structures in recent versions.
        
        Ideally for generation, we generate the TLE string first, then parse it.
        """
        # This function might be better placed in the generation module
        pass

    @staticmethod
    def get_orbital_elements(satellite):
        """
        Extracts orbital elements from a Satrec object.
        """
        # Note: Satrec objects store these in internal units (radians, etc.)
        # sgp4 2.x uses lowercase attribute names
        return {
            'inclination': satellite.inclo,
            'raan': satellite.nodeo,
            'eccentricity': satellite.ecco,
            'arg_perigee': satellite.argpo,
            'mean_anomaly': satellite.mo,
            'mean_motion': satellite.no_kozai
        }
