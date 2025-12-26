# Drug Usage Predictor - PyTorch Implementation

Linear Regression implemented using PyTorch framework for predicting drug usage.

## Project Structure

```
PyTorchImplementation/
├── src/
│   ├── data/
│   │   ├── synthetic_data_generator.py
│   │   └── preprocessing.py
│   ├── models/
│   │   └── linear_regression_pytorch.py
│   └── utils/
│       └── visualization.py
├── reports/figures/
├── main_pytorch.py
└── requirements.txt
```

## Features
- Linear Regression using PyTorch nn.Module
- SGD optimizer with backpropagation
- Automatic differentiation
- GPU support (if available)
- Same synthetic data as NumPy implementation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:
```bash
python main_pytorch.py
```

## PyTorch Implementation Details

- **Model**: `nn.Linear` layer for linear regression
- **Loss**: MSELoss (Mean Squared Error)
- **Optimizer**: SGD (Stochastic Gradient Descent)
- **Training**: 10,000 epochs with backpropagation

## Comparison with NumPy Implementation

| Feature | NumPy (Manual) | PyTorch |
|---------|----------------|---------|
| Implementation | Manual gradient descent | Automatic differentiation |
| Speed | Fast for small data | Faster with GPU |
| Flexibility | Full control | Framework abstraction |
| Code complexity | More code | Less code |
