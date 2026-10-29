# Zero Overhead Complexity Estimation for Adaptive Dynamic Inference Before Execution in Edge AI

This repository provides the Python implementation of Zero Overhead Complexity Estimation (ZOCE) for adaptive dynamic inference in Edge AI.

ZOCE makes dynamic model routing decisions before executing any neural network backbone. Using ten lightweight image quality and complexity descriptors computed directly from raw pixel inputs, a quadratic gate estimates the accuracy gain of escalating from a lightweight backbone to a heavy backbone under dynamic compute budgets.

## Code Structure

- `zoce_features.py`: Implementation of the 10 zero-overhead image quality and complexity descriptors (luminance, entropy, variance, gradient density, noise sigma, Laplacian energy, gradient tail weight, two-scale ratio, chroma spread, block ratio).
- `01_run_backbones.py`: Constructs the evaluation stream (clean CIFAR-100 mixed with CIFAR-100-C corruptions) and evaluates frozen model backbones.
- `02_gate_experiments.py`: Trains the gain-aware quadratic gates and evaluates routing policies (ZOCE-R, ZOCE-T, confidence cascade, learned CNN gate, data-blind switching, and baselines).
- `03_budget_controller.py`: Online primal-dual threshold controller adapting to non-stationary request streams and telemetry constraints (battery/thermal derating).
- `04_export_figures_tables.py`: Aggregates results across splits and formats evaluation metrics and summary tables.
- `05_measure_latency.py`: Single-thread execution latency benchmarking for backbones and descriptor extraction.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- thop
- `pytorch_cifar_models`
- `robustbench`

## Usage

Execute the pipeline stages in order:

```bash
# 1. Generate stream and extract backbone predictions
python 01_run_backbones.py

# 2. Train gates and evaluate routing policies
python 02_gate_experiments.py

# 3. Evaluate the online budget controller
python 03_budget_controller.py

# 4. Aggregate results and export tables
python 04_export_figures_tables.py

# 5. Benchmark inference latency
python 05_measure_latency.py
```
