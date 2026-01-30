"""
Neural network model definition and training logic.
Optimized for RTX GPU with mixed precision training.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ============== GPU OPTIMIZATION ==============
def setup_gpu():
    """Configure TensorFlow for optimal GPU utilization."""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"🎮 Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
        
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision training (FP16) for RTX GPUs
            # This uses Tensor Cores for ~2-3x speedup
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("✅ Mixed Precision (FP16) enabled - Using Tensor Cores for faster training!")
            
            # Enable XLA JIT compilation for additional speedup
            tf.config.optimizer.set_jit(True)
            print("✅ XLA JIT compilation enabled!")
            
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
    else:
        print("⚠️ No GPU found. Training will use CPU.")
    
    return len(gpus) > 0

# Run GPU setup on module load
HAS_GPU = setup_gpu()
# =============================================

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import MODEL_CONFIG, TRAINING_CONFIG, PATHS, DATA_DIR, FEATURE_CONFIG
from src.utils.feature_extraction import FeatureExtractor


class CollisionModel:
    def __init__(self):
        self.model = self.build_model()
        self.scaler = StandardScaler()
        
    def build_model(self):
        """Builds the multi-output neural network optimized for GPU."""
        input_layer = layers.Input(shape=(MODEL_CONFIG['input_dim'],))
        
        # Shared layers with GPU-optimized settings
        x = input_layer
        for units in MODEL_CONFIG['hidden_layers']:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(MODEL_CONFIG['dropout_rate'])(x)
            
        # Output branches
        # 1. Collision Probability (Binary Classification)
        # Note: For mixed precision, sigmoid output needs float32 for numerical stability
        collision_output = layers.Dense(1, activation='sigmoid', dtype='float32', name='collision_output')(x)
        
        # 2. Time to Collision (Regression)
        time_output = layers.Dense(1, activation='linear', dtype='float32', name='time_output')(x)
        
        # 3. Collision Location (Regression - 3 coords: lat, lon, alt)
        location_output = layers.Dense(3, activation='linear', dtype='float32', name='location_output')(x)
        
        model = models.Model(inputs=input_layer, outputs=[collision_output, time_output, location_output])
        
        # Use legacy Adam optimizer compatible with mixed precision
        optimizer = optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss={
                'collision_output': 'binary_crossentropy',
                'time_output': 'mse',
                'location_output': 'mse'
            },
            loss_weights={
                'collision_output': 1.0,
                'time_output': 0.5,
                'location_output': 0.3
            },
            metrics={
                'collision_output': 'accuracy',
                'time_output': 'mae',
                'location_output': 'mae'
            }
        )
        return model

    def train(self, synthetic_collision_path, synthetic_safe_path):
        """Loads data, processes it, and trains the model with GPU acceleration."""
        
        print("\n" + "="*50)
        print("   SATELLITE COLLISION MODEL TRAINING")
        print("="*50)
        
        if HAS_GPU:
            print("🚀 GPU Training Mode Active!")
        else:
            print("💻 CPU Training Mode")
        
        print("\nLoading data...")
        # Load JSONs
        with open(synthetic_collision_path, 'r') as f:
            collision_data = json.load(f)
        with open(synthetic_safe_path, 'r') as f:
            safe_data = json.load(f)
            
        all_data = collision_data + safe_data
        
        X = []
        y_collision = []
        y_time = []
        y_location = []  # [lat, lon, alt]
        
        print(f"Extracting features from {len(all_data)} samples...")
        
        for item in tqdm(all_data, desc="Feature extraction"):
            tle_a = item['satellite_a']
            tle_b = item['satellite_b']
            
            try:
                feats, _ = FeatureExtractor.extract_features(
                    tle_a['tle_line1'], tle_a['tle_line2'],
                    tle_b['tle_line1'], tle_b['tle_line2']
                )
                
                if np.all(feats == 0):
                    continue
                    
                X.append(feats)
                y_collision.append(1 if item['collision'] else 0)
                
                if item['collision']:
                    y_time.append(item['collision_time_days'])
                    loc = item['collision_location']
                    y_location.append([loc.get('latitude', 0), loc.get('longitude', 0), loc.get('altitude_km', 0)])
                else:
                    y_time.append(FEATURE_CONFIG['propagate_window_days'] * 2)
                    y_location.append([0, 0, 0])
                    
            except Exception as e:
                continue

        # Convert to numpy arrays with float32 for GPU efficiency
        X = np.array(X, dtype=np.float32)
        y_collision = np.array(y_collision, dtype=np.float32)
        y_time = np.array(y_time, dtype=np.float32)
        y_location = np.array(y_location, dtype=np.float32)
        
        print(f"✓ Successfully extracted {len(X)} samples")
        
        # Split data
        print("Splitting data...")
        (X_train, X_val, 
         y_col_train, y_col_val,
         y_time_train, y_time_val,
         y_loc_train, y_loc_val) = train_test_split(
            X, y_collision, y_time, y_location,
            test_size=TRAINING_CONFIG['validation_split'],
            stratify=y_collision,
            random_state=42
        )
        
        # Scale
        print("Fitting scaler...")
        X_train = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val = self.scaler.transform(X_val).astype(np.float32)
        
        # Save scaler
        os.makedirs(os.path.dirname(PATHS['scaler_path']), exist_ok=True)
        joblib.dump(self.scaler, PATHS['scaler_path'])
        
        # Prepare targets
        train_targets = {
            'collision_output': y_col_train,
            'time_output': y_time_train,
            'location_output': y_loc_train
        }
        val_targets = {
            'collision_output': y_col_val,
            'time_output': y_time_val,
            'location_output': y_loc_val
        }
        
        # Create TensorFlow datasets for better GPU utilization
        print("Creating optimized data pipelines...")
        
        # Prefetch and cache for GPU efficiency
        AUTOTUNE = tf.data.AUTOTUNE
        batch_size = TRAINING_CONFIG['batch_size']
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, train_targets))
        train_dataset = train_dataset.shuffle(buffer_size=10000)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, val_targets))
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(AUTOTUNE)
        
        # Callbacks
        os.makedirs(os.path.dirname(PATHS['model_path']), exist_ok=True)
        
        checkpoint_cb = callbacks.ModelCheckpoint(
            PATHS['model_path'], 
            save_best_only=TRAINING_CONFIG['save_best_only'], 
            monitor='val_collision_output_accuracy', 
            mode='max',
            verbose=1
        )
        early_stop_cb = callbacks.EarlyStopping(
            patience=TRAINING_CONFIG['early_stopping_patience'], 
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr_cb = callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=TRAINING_CONFIG['reduce_lr_factor'],
            patience=TRAINING_CONFIG['reduce_lr_patience'],
            verbose=1,
            min_lr=1e-7
        )
        
        # TensorBoard for monitoring (optional)
        tensorboard_cb = callbacks.TensorBoard(
            log_dir=os.path.join(PATHS['model_path'].replace('.h5', '_logs')),
            histogram_freq=1,
            profile_batch='10,20'  # Profile GPU usage on batches 10-20
        )
        
        print(f"\n🏋️ Starting training:")
        print(f"   Train samples: {len(X_train)}")
        print(f"   Val samples:   {len(X_val)}")
        print(f"   Batch size:    {batch_size}")
        print(f"   Epochs:        {TRAINING_CONFIG['epochs']}")
        print("="*50 + "\n")
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=TRAINING_CONFIG['epochs'],
            callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb, tensorboard_cb],
            verbose=1
        )
        
        # Save final model
        self.model.save(PATHS['model_path'])
        print(f"\n✅ Model saved to {PATHS['model_path']}")
        
        # Print final metrics
        print("\n" + "="*50)
        print("   TRAINING COMPLETE!")
        print("="*50)
        best_acc = max(history.history.get('val_collision_output_accuracy', [0]))
        print(f"   Best Validation Accuracy: {best_acc*100:.2f}%")
        print("="*50)
        
        return history


if __name__ == "__main__":
    # Test run
    model = CollisionModel()
