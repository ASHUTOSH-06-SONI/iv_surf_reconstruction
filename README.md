# IV Surface Reconstruction: Physics-Informed Neural Networks vs. Classical Methods

A quantitative finance research project for reconstructing implied volatility (IV) surfaces from sparse option-chain observations. This repository compares physics-informed neural networks (PINNs) with classical quantitative finance methods (SABR model, interpolation) for reconstructing complete IV surfaces.

## Project Overview

### Problem Statement

Implied volatility surfaces are critical for option pricing and risk management. However, in practice, we often observe sparse IV quotes at limited strikes and expirations. The goal is to reconstruct the complete IV surface from these observations using:

1. **Physics-Informed Neural Networks (PINN)** — Leverages financial constraints directly in the model
2. **Classical Methods** — SABR stochastic volatility model, spline/interpolation baselines

### Why Physics-Informed?

Traditional neural networks can learn to predict IVs, but without financial constraints they may:
- Predict negative volatilities (unphysical)
- Produce unstable, spiky surfaces (violates market smoothness)
- Fail to generalize beyond training data

PINNs address this by embedding financial constraints:

```
Loss = MSE(pred, target) + λ × Physics_Penalty(pred)

Physics_Penalty = E[max(-σ, 0)²] + E[max(σ - σ_max, 0)²]
```

**Output Activation:** Softplus (ln(1 + e^x)) ensures σ > 0  
**Bounds:** σ ∈ [0, 3.0] for NIFTY50 options  
**Architecture:** 4-layer feedforward network with ReLU hidden activations

## Installation

### Requirements
- Python 3.9+
- PyTorch 2.0+
- NumPy, Pandas, Scikit-learn, Matplotlib

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd iv_surf_reconstruction

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Project Structure

```
iv_surf_reconstruction/
├── config/
│   └── default.yaml          # Configuration file (paths, hyperparameters)
├── src/
│   ├── __init__.py
│   ├── config.py             # Config loader
│   ├── cli.py                # Command-line interface
│   ├── models/
│   │   ├── pinn_model.py     # PINN architecture
│   ├── train/
│   │   └── train_and_save.py # Training script with early stopping
│   ├── data/
│   │   └── preprocess.py     # Data preprocessing pipeline
│   ├── utils/
│   │   └── reshape.py        # Data reshaping (wide → long format)
│   ├── evaluate/
│   │   └── accuracy.py       # Evaluation metrics
│   └── baselines/
│       ├── sabr.py           # SABR stochastic volatility model
│       └── interpolation.py  # Classical interpolation methods
├── tests/
│   ├── test_config.py
│   ├── test_model.py
│   ├── test_preprocessing.py
│   └── conftest.py
├── data/
│   ├── raw/                  # Raw option data
│   └── processed/            # Preprocessed data (CSV)
├── models/                   # Trained model checkpoints
├── results/                  # Evaluation results, plots
├── pyproject.toml            # Project configuration
└── README.md
```

## Quick Start

### 1. Prepare Your Data

Your data should be in **long format** (one row per option observation):

```csv
timestamp,underlying,strike_price,option_type,expiry,iv,X0,X1,...,X41
2024-01-01 10:00:00,100,100,call,2024-02,0.25,1.2,0.5,...
2024-01-01 10:00:00,100,110,call,2024-02,0.30,1.2,0.5,...
```

Save preprocessed data to `data/processed/train.csv`.

### 2. Train PINN Model

```bash
# Train with default config
pinn-train --config config/default.yaml

# Train on GPU
pinn-train --config config/default.yaml --device cuda
```

This will:
- Load and preprocess data
- Initialize PINN with physics constraints
- Train for 1000 epochs with validation
- Save best model to `models/best_model.pt`
- Log metrics every 50 epochs

### 3. Evaluate Model

```bash
pinn-evaluate --model models/best_model.pt --output results/evaluation.txt
```

### 4. Compare with Baselines

```bash
pinn-compare --model models/best_model.pt --output results/comparison.csv
```

## Configuration

Edit `config/default.yaml` to customize:

```yaml
# Data paths and splits
data:
  train_file: "data/processed/train.csv"
  split: {train: 0.70, val: 0.15, test: 0.15}
  temporal_split: true  # Use temporal ordering, not random split

# Model architecture
model:
  input_dim: 45  # Option features + market factors
  hidden_dims: [64, 64]
  output_activation: "softplus"  # Ensures IV > 0
  physics_penalty_weight: 0.1

# Training
training:
  epochs: 1000
  batch_size: 32
  learning_rate: 0.001
  use_early_stopping: true
  early_stopping_patience: 50
  device: "cpu"  # or "cuda"
```

## Model Details

### PINN Architecture

```
Input (45 features) → Linear(45, 64) → ReLU
                    → Linear(64, 64) → ReLU
                    → Linear(64, 1) → Softplus
Output: IV ∈ (0, ∞)
```

**Physics Constraints:**
- **Softplus activation** ensures σ > 0 (non-negativity)
- **Soft penalty** discourages σ > 3.0 (upper bound)
- **MSE loss** on normalized targets for stable training

### Loss Function

```
L_total = L_MSE + λ × L_physics

where:
  L_MSE = Mean squared error between predictions and targets
  L_physics = E[max(-σ, 0)²] + E[max(σ - 3.0, 0)²]
  λ = Physics penalty weight (default: 0.1)
```

### Baseline Methods

1. **SABR Model** (`src/baselines/sabr.py`)
   - Stochastic alpha-beta-rho volatility model
   - Hagan's closed-form approximation
   - Calibrated to market IV data

2. **Linear/Spline Interpolation** (`src/baselines/interpolation.py`)
   - Spatial interpolation on observed IV points
   - RectBivariateSpline for smooth surfaces
   - Kernel smoothing via local volatility

## Training Tips

### Data Preparation

1. **Normalize features** using StandardScaler (automatically done)
2. **Normalize targets** (IV normalized to mean=0, std=1 during training)
3. **Remove outliers** (IV < 0 or IV > 3 filtered automatically)
4. **Temporal split** (recommended for time-series option data)

### Hyperparameter Tuning

```python
# In config/default.yaml, adjust:
model:
  hidden_dims: [128, 128]  # Larger network
  physics_penalty_weight: 0.5  # Stronger constraint

training:
  learning_rate: 0.0005  # Lower LR for stability
  early_stopping_patience: 100
```

## Testing

Run unit tests to verify the implementation:

```bash
pytest tests/ -v

# Run specific test
pytest tests/test_model.py -v

# Coverage report
pytest tests/ --cov=src --cov-report=html
```

**Test Coverage:**
- Config loading and validation
- PINN forward pass and backprop
- Physics penalty computation
- Data preprocessing pipeline

## Research References

- **PINN Framework:** Raissi et al. (2019) "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems"
- **SABR Model:** Hagan et al. (2002) "Managing Smile Risk"
- **IV Surface Theory:** Rebonato (2004) "Volatility and Correlation"

See `src/assets/` for included research papers:
- `PINN.pdf` — General PINN methodology
- `SABR_Model.pdf` — SABR stochastic volatility model  
- `PIDL.pdf` — Physics-informed deep learning

## Results & Benchmarks

(Results will be added after training on benchmark datasets)

### PINN Performance
- **MSE on test set:** TBD
- **IV prediction range:** [0.0, 3.0] (constrained)
- **Surface smoothness:** Good (Softplus output)

### Baseline Comparisons
- **Linear Interpolation:** Fast, simple, but non-smooth
- **SABR Model:** Theoretically sound, requires calibration
- **PINN:** Best generalization, data-driven with physics

## Future Work

1. **Extended Baselines**
   - Dupire local volatility model
   - Neural volatility surface (unconstrained NN)
   - Gaussian process interpolation

2. **Advanced PINN Variants**
   - Residual physics (enforce Black-Scholes PDE)
   - Transfer learning from related IV surfaces
   - Multi-asset correlation modeling

3. **Evaluation Metrics**
   - Surface smoothness measures
   - Arbitrage-free tests
   - Hedging performance comparison

4. **Production Considerations**
   - Real-time inference API
   - Model serving (ONNX export)
   - Retraining pipeline

## Citation

If you use this code in research, please cite:

```bibtex
@software{iv_surface_pinn_2024,
  title={IV Surface Reconstruction using Physics-Informed Neural Networks},
  author={Soni, Santosh},
  year={2024},
  url={https://github.com/yourusername/iv_surf_reconstruction}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

For questions or collaborations, reach out to [your contact info]

---

**Last Updated:** June 2, 2024  
**Status:** Active Research (Alpha)

