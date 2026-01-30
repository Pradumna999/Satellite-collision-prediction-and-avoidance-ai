"""
Module for analyzing TLE data to detect collisions and generate reports.
"""
import os
import sys
import json
import numpy as np
import datetime
import joblib
from tensorflow.keras.models import load_model, Model
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import PATHS, PREDICTION_CONFIG, MANEUVER_CONFIG
from src.utils.feature_extraction import FeatureExtractor
from src.prediction.maneuver_calculator import ManeuverCalculator
from src.utils.tle_utils import TLEHandler

class CollisionAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.maneuver_calc = ManeuverCalculator(MANEUVER_CONFIG)
        
    def load_resources(self):
        """Loads trained model and scaler."""
        if not os.path.exists(PATHS['model_path']):
            raise FileNotFoundError("Model not found. Please train the model first.")
        if not os.path.exists(PATHS['scaler_path']):
            raise FileNotFoundError("Scaler not found. Please train the model first.")
            
        print("Loading model and scaler...")
        # Load model without compiling to avoid Keras 3.x serialization issues
        # We only need the model for inference, not training
        self.model = load_model(PATHS['model_path'], compile=False)
        self.scaler = joblib.load(PATHS['scaler_path'])
        
    def analyze_file(self, tle_file_path):
        """
        Analyzes a JSON file containing TLEs.
        Returns report string.
        """
        with open(tle_file_path, 'r') as f:
            data = json.load(f)
            
        satellites = data.get('satellites', [])
        if len(satellites) < 2:
            return "Not enough satellites to analyze."
            
        results = []
        safe_pairs_count = 0
        
        # Generate pairs
        # For N satellites, N*(N-1)/2 pairs
        pairs = []
        for i in range(len(satellites)):
            for j in range(i+1, len(satellites)):
                pairs.append((satellites[i], satellites[j]))
                
        print(f"Analyzing {len(pairs)} satellite pairs...")
        
        for sat_a, sat_b in tqdm(pairs):
            # Extract features
            try:
                feats, meta = FeatureExtractor.extract_features(
                    sat_a['tle_line1'], sat_a['tle_line2'],
                    sat_b['tle_line1'], sat_b['tle_line2']
                )
                
                # Scale
                feats_scaled = self.scaler.transform(feats.reshape(1, -1))
                
                # Predict
                preds = self.model.predict(feats_scaled, verbose=0)
                prob_collision = preds[0][0][0] # collision_output
                time_to_col = preds[1][0][0]    # time_output
                loc_col = preds[2][0]           # location_output
                
                if prob_collision > PREDICTION_CONFIG['collision_prob_threshold']:
                    # Collision detected
                    collision_time = datetime.datetime.now() + datetime.timedelta(days=float(time_to_col))
                    
                    # Calculate maneuver
                    # We need Satrec object for A
                    sat_obj_a = TLEHandler.parse_tle(sat_a['tle_line1'], sat_a['tle_line2'])
                    maneuver = self.maneuver_calc.calculate_maneuver(
                        sat_obj_a, collision_time, meta.get('current_dist', 0)
                    )
                    
                    results.append({
                        'sat_a': sat_a,
                        'sat_b': sat_b,
                        'probability': prob_collision,
                        'time_days': time_to_col,
                        'location': loc_col,
                        'maneuver': maneuver,
                        'current_dist': meta.get('current_dist', 0)
                    })
                else:
                    safe_pairs_count += 1
                    
            except Exception as e:
                print(f"Error analyzing pair {sat_a['name']} vs {sat_b['name']}: {e}")
                continue
                
        return self._generate_report(results, safe_pairs_count, len(pairs))
        
    def _generate_report(self, collisions, safe_count, total_pairs):
        """Formats the output report."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = []
        report.append("=== SATELLITE COLLISION PREDICTION RESULTS ===")
        report.append(f"Analysis Date: {timestamp}")
        report.append(f"Total Pairs Analyzed: {total_pairs}")
        report.append(f"Collisions Detected: {len(collisions)}")
        report.append("")
        
        if collisions:
            report.append("⚠️ COLLISION WARNINGS ⚠️")
            report.append("")
            
            for i, col in enumerate(collisions):
                report.append(f"COLLISION #{i+1}")
                report.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                report.append(f"Satellite Pair: {col['sat_a']['name']} vs {col['sat_b']['name']}")
                report.append(f"Collision Probability: {col['probability']*100:.1f}%")
                report.append(f"Time to Collision: {col['time_days']:.1f} days")
                report.append("Predicted Collision Location:")
                report.append(f"  Latitude:  {col['location'][0]:.4f}°")  # Placeholder logic for XYZ->LatLon not implemented fully
                report.append(f"  Longitude: {col['location'][1]:.4f}°") 
                report.append(f"  Altitude:  {col['location'][2]:.2f} km")
                report.append("")
                report.append("Current Telemetry:")
                report.append(f"  Current Separation: {col['current_dist']:.2f} km")
                report.append("")
                report.append("🛰️ RECOMMENDED AVOIDANCE MANEUVER:")
                m = col['maneuver']
                report.append(f"  Type: {m['type']}")
                report.append(f"  Direction: {m['direction']}")
                report.append(f"  Delta-V Required: {m['delta_v']} m/s")
                report.append(f"  Execute Time: {m['execute_time']}")
                report.append(f"  Success Probability: {m['success_probability']}%")
                report.append("")
                report.append("  Precautions:")
                for p in m['precautions']:
                    report.append(f"    • {p}")
                report.append("")
                report.append("")
        
        report.append(f"✓ SAFE PAIRS ({safe_count})")
        # Could list safe pairs but might be too long
        
        return "\n".join(report)

    def save_report(self, report_text, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Report saved to {path}")

if __name__ == "__main__":
    # Test
    # Needs model to run
    pass
