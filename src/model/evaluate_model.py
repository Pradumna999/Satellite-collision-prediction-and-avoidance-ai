"""
Module for evaluating the trained model's performance.
"""
import os
import sys
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import PATHS, SYNTHETIC_DATA_CONFIG, MODEL_CONFIG
from src.utils.feature_extraction import FeatureExtractor


class ModelEvaluator:
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def load_resources(self):
        if not os.path.exists(PATHS['model_path']):
            raise FileNotFoundError("Model not found. Train first.")
        if not os.path.exists(PATHS['scaler_path']):
            raise FileNotFoundError("Scaler not found. Train first.")
            
        # Load model without compiling to avoid Keras 3.x serialization issues
        self.model = load_model(PATHS['model_path'], compile=False)
        self.scaler = joblib.load(PATHS['scaler_path'])
        
    def evaluate(self, num_samples=500):
        """
        Generates a fresh test set and evaluates the model.
        """
        print(f"Generating temporary test set ({num_samples} samples)...")
        
        # Import the generation functions with correct names
        from src.data_generation.generate_synthetic_data import generate_collision_scenario, generate_safe_scenario
        import datetime
        
        epoch = datetime.datetime.now()
        n = num_samples // 2
        
        test_data = []
        
        print("Generating test scenarios...")
        for i in tqdm(range(n), desc="Collision scenarios"):
            sat_id_a = 90000 + i * 2
            sat_id_b = 90001 + i * 2
            c = generate_collision_scenario(sat_id_a, sat_id_b, epoch)
            test_data.append(c)
            
        for i in tqdm(range(n), desc="Safe scenarios"):
            sat_id_a = 95000 + i * 2
            sat_id_b = 95001 + i * 2
            s = generate_safe_scenario(sat_id_a, sat_id_b, epoch)
            test_data.append(s)
            
        print(f"Extracting features for {len(test_data)} samples...")
        X = []
        y_true_col = []
        y_true_time = []
        
        for item in tqdm(test_data, desc="Feature extraction"):
            try:
                tle_a = item['satellite_a']
                tle_b = item['satellite_b']
                feats, _ = FeatureExtractor.extract_features(
                    tle_a['tle_line1'], tle_a['tle_line2'],
                    tle_b['tle_line1'], tle_b['tle_line2']
                )
                
                if np.all(feats == 0):
                    continue
                    
                X.append(feats)
                y_true_col.append(1 if item['collision'] else 0)
                
                # Time only defined for collisions ideally, but model outputs it always
                if item['collision']:
                    y_true_time.append(item['collision_time_days'])
                else:
                    y_true_time.append(np.nan)  # Ignore safe for time metric
            except Exception as e:
                continue
                
        X = np.array(X)
        y_true_col = np.array(y_true_col)
        y_true_time = np.array(y_true_time)
        
        print(f"Valid test samples: {len(X)}")
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        print("Running predictions...")
        preds = self.model.predict(X_scaled, verbose=0)
        
        pred_prob_col = preds[0]
        pred_time = preds[1].flatten()
        
        # Binary Classification Metrics
        y_pred_col = (pred_prob_col > 0.5).astype(int).flatten()
        
        acc = accuracy_score(y_true_col, y_pred_col)
        prec = precision_score(y_true_col, y_pred_col, zero_division=0)
        rec = recall_score(y_true_col, y_pred_col, zero_division=0)
        f1 = f1_score(y_true_col, y_pred_col, zero_division=0)
        cm = confusion_matrix(y_true_col, y_pred_col)
        
        # Regression Metrics (Time) - only for true collisions
        mask = ~np.isnan(y_true_time)
        if np.sum(mask) > 0:
            mae_time = mean_absolute_error(y_true_time[mask], pred_time[mask])
        else:
            mae_time = 0.0
            
        print("\n" + "="*30)
        print("   MODEL EVALUATION REPORT   ")
        print("="*30)
        print(f"Test Samples: {len(X)}")
        print(f"Accuracy:     {acc:.4f}")
        print(f"Precision:    {prec:.4f}")
        print(f"Recall:       {rec:.4f}")
        print(f"F1 Score:     {f1:.4f}")
        print(f"Time MAE:     {mae_time:.4f} days")
        print("-" * 30)
        print("Confusion Matrix:")
        print(cm)
        print("="*30)
        print("\nInterpretation:")
        if acc > 0.95:
            print("✅ Model is performing EXCELLENTLY (>95% Accuracy).")
        elif acc > 0.85:
            print("⚠️ Model is performing GOOD (>85% Accuracy).")
        elif acc > 0.70:
            print("⚠️ Model is performing FAIR (>70% Accuracy). May need more training data.")
        else:
            print("❌ Model performance is SUBOPTIMAL. Check data or training parameters.")
            
if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.load_resources()
    evaluator.evaluate()
