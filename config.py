"""
Configuration parameters for the Satellite Collision Prediction System.
Optimized for MAXIMUM ACCURACY with 400,000 total samples.
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Data Generation Configuration - SCALED TO 2 LAKH EACH
SYNTHETIC_DATA_CONFIG = {
    'num_collision_samples': 200000,      # 2 Lakh collision scenarios
    'num_non_collision_samples': 200000,  # 2 Lakh non-collision scenarios
    'collision_time_window': 7,  # days
    'min_collision_dist_km': 5.0,
    'safe_separation_dist_km': 50.0,
    'altitude_range_km': (300, 2000),
    'inclination_range_deg': (0, 180),
}

# Feature Extraction Configuration
FEATURE_CONFIG = {
    'propagate_window_days': 7,
    'time_steps': 10,  # Number of points to sample for closest approach refinement
}

# Model Architecture Configuration - OPTIMIZED FOR MAX ACCURACY
MODEL_CONFIG = {
    'input_dim': 42,
    'hidden_layers': [512, 256, 128, 64, 32],  # Deeper network for better feature learning
    'dropout_rate': 0.25,  # Slightly reduced dropout for larger dataset
    'learning_rate': 0.0005,  # Lower LR for better convergence
}

# Training Configuration - OPTIMIZED FOR MAX ACCURACY
TRAINING_CONFIG = {
    'batch_size': 128,  # Larger batch for better gradient estimates
    'epochs': 150,  # More epochs to allow full convergence
    'validation_split': 0.15,  # Less validation, more training data
    'early_stopping_patience': 20,  # More patience for deeper networks
    'reduce_lr_patience': 10,
    'reduce_lr_factor': 0.5,
    'save_best_only': True,
}

# Prediction & Maneuver Configuration
PREDICTION_CONFIG = {
    'collision_prob_threshold': 0.5,
    'analysis_window_days': 14,
}

MANEUVER_CONFIG = {
    'min_delta_v': 0.5,  # m/s
    'max_delta_v': 50.0,  # m/s
    'safety_margin_km': 10.0,
    'lead_time_hours': 48,
    'fuel_weight': 1.0,  # Weight for fuel optimization cost profile
}

# Paths for artifacts
PATHS = {
    'synthetic_collision': os.path.join(DATA_DIR, 'synthetic', 'collision_data.json'),
    'synthetic_safe': os.path.join(DATA_DIR, 'synthetic', 'non_collision_data.json'),
    'real_data': os.path.join(DATA_DIR, 'real', 'real_data.json'),
    'model_path': os.path.join(MODELS_DIR, 'trained', 'collision_model.h5'),
    'scaler_path': os.path.join(MODELS_DIR, 'trained', 'feature_scaler.pkl'),
    'results_file': os.path.join(RESULTS_DIR, 'result.txt'),
}
