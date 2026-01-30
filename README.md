# Satellite Collision Prediction System

A production-ready machine learning system that predicts satellite collisions using Two-Line Element (TLE) data, recommends avoidance maneuvers, and provides comprehensive collision analysis reports.

## 🚀 Features

- **Synthetic Data Generation**: Generates 100,000+ realistic TLE datasets for training.
- **Deep Learning Model**: Multi-output neural network predicting collision probability, time, and location.
- **Collision Analysis**: Efficiently processes satellite pairs to detect conjunctions.
- **Avoidance Maneuvers**: Calculates optimal delta-v and timing for collision avoidance using orbital mechanics.
- **Comprehensive Reports**: Generates detailed text reports with maneuver recommendations.

## 📋 Installation

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Usage

Run the main interactive CLI:

```bash
python main.py
```

### Menu Options

1. **Generate Synthetic TLE Data**: targeted 100k samples (50k collision, 50k safe).
2. **Train ML Model**: Trains the neural network on generated data.
3. **Predict Collisions from Real Data**: Analyzes `data/real/real_data.json` and outputs `results/result.txt`.

## 📂 Project Structure

- `src/data_generation/`: TLE generator.
- `src/model/`: Neural network model (Keras).
- `src/utils/`: Feature extraction and TLE propagation (SGP4).
- `src/prediction/`: Collision analyzer and maneuver calculator.
- `data/`: Storage for synthetic and real data.
- `results/`: Output reports.

## 📊 Model Details

The model uses a dense neural network with:
- **Input**: 42 features (Orbital Elements, Relative State Vectors)
- **Outputs**: 
  - Collision Probability (Sigmoid)
  - Time to Collision (Linear regression)
  - Collision Location (Linear regression)

## ⚠️ Disclaimer
This tool is for educational and analysis purposes. Always verify results with certified space traffic management providers before operational maneuvers.
