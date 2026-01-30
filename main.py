import os
import sys
import time

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import PATHS, SYNTHETIC_DATA_CONFIG, DATA_DIR
from src.data_generation.generate_synthetic_data import generate_dataset
from src.model.collision_model import CollisionModel
from src.prediction.collision_analyzer import CollisionAnalyzer

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("="*50)
    print("   SATELLITE COLLISION PREDICTION SYSTEM")
    print("="*50)
    print()

def main_menu():
    while True:
        clear_screen()
        print_header()
        print("1. Generate Synthetic TLE Data (Collision + Non-Collision)")
        print("2. Train ML Model on Synthetic Data")
        print("3. Test/Evaluate Model Performance")
        print("4. Predict Collisions from Real Data (real_data.json)")
        print("5. Exit")
        print()
        
        choice = input("Select an option (1-5): ")
        
        if choice == '1':
            print("\nGenerating data...")
            # We assume the background process might have done this, but user can re-run
            try:
                generate_dataset(
                    SYNTHETIC_DATA_CONFIG['num_collision_samples'],
                    SYNTHETIC_DATA_CONFIG['num_non_collision_samples'],
                    DATA_DIR
                )
                print("\nData generation complete!")
            except Exception as e:
                print(f"\nError: {e}")
            input("\nPress Enter to continue...")
            
        elif choice == '2':
            print("\nTraining model...")
            try:
                model_wrapper = CollisionModel()
                if not os.path.exists(PATHS['synthetic_collision']):
                    print("Error: Synthetic data not found. Run option 1 first.")
                else:
                    model_wrapper.train(PATHS['synthetic_collision'], PATHS['synthetic_safe'])
                    print("\nTraining complete!")
            except Exception as e:
                print(f"\nError: {e}")
            input("\nPress Enter to continue...")
            
        elif choice == '3':
            print("\nEvaluating model performance...")
            try:
                from src.model.evaluate_model import ModelEvaluator
                evaluator = ModelEvaluator()
                evaluator.load_resources()
                evaluator.evaluate(num_samples=500) # Small batch for quick interactive test
            except Exception as e:
                print(f"Evaluation failed: {e}")
            input("\nPress Enter to continue...")
            
        elif choice == '4':
            print("\nPredicting collisions...")
            try:
                analyzer = CollisionAnalyzer()
                analyzer.load_resources()
                
                # Check for real data
                if not os.path.exists(PATHS['real_data']):
                    print(f"Error: {PATHS['real_data']} not found. Creating a dummy file for testing...")
                    # Create dummy real data
                    dummy_data = {
                        "satellites": [
                            {"name": "ISS", "tle_line1": "1 25544U 98067A   23001.00000000  .00016717  00000-0  10270-3 0  9999", "tle_line2": "2 25544  51.6400 208.9163 0006703  69.9862  25.2906 15.54225995123456"},
                            {"name": "STARLINK-1001", "tle_line1": "1 44713U 19074A   23001.00000000  .00000000  00000-0  00000-0 0  9999", "tle_line2": "2 44713  53.0000   0.0000 0001000   0.0000  69.0000 15.00000000    1"},
                             {"name": "DEBRIS-X", "tle_line1": "1 99999U 19074A   23001.00000000  .00000000  00000-0  00000-0 0  9999", "tle_line2": "2 99999  51.6400 208.9163 0006703  69.9862  25.2906 15.54225995123456"} # Same as ISS
                        ]
                    }
                    os.makedirs(os.path.dirname(PATHS['real_data']), exist_ok=True)
                    import json
                    with open(PATHS['real_data'], 'w') as f:
                        json.dump(dummy_data, f, indent=2)
                
                report = analyzer.analyze_file(PATHS['real_data'])
                print("\n" + report)
                analyzer.save_report(report, PATHS['results_file'])
                
            except Exception as e:
                print(f"\nError details: {e}")
                import traceback
                traceback.print_exc()
            input("\nPress Enter to continue...")
            
        elif choice == '5':
            print("Exiting...")
            break

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nExiting...")
