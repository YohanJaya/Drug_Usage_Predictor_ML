# Drug Usage Predictor ML

Linear Regression implementations for predicting drug usage using multiple features.

## Project Structure

This project contains two implementations:

### 1. **BasicImplementation/** - Manual NumPy Implementation
- Pure NumPy with manual gradient descent
- Educational: See how gradient descent works from scratch
- Full control over the algorithm

### 2. **PyTorchImplementation/** - PyTorch Framework
- Uses PyTorch nn.Module
- Automatic differentiation
- Production-ready with GPU support

```
drugUsagePredictorML/
├── BasicImplementation/          # Manual gradient descent with NumPy
│   ├── src/
│   │   ├── data/
│   │   ├── models/
│   │   └── utils/
│   ├── main.py
│   └── README.md
│
├── PyTorchImplementation/        # PyTorch framework implementation
│   ├── src/
│   │   ├── data/
│   │   ├── models/
│   │   └── utils/
│   ├── main_pytorch.py
│   └── README.md
│
└── requirements.txt              # Shared dependencies
```

## Features
- Linear Regression with multiple features (4 features)
- Synthetic data generation (365 days)
- Feature normalization
- Model evaluation metrics (MSE, RMSE, R², MAE)
- Visualization tools (cost history, predictions, residuals, feature importance)

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

For PyTorch implementation, also install:
```bash
cd PyTorchImplementation
pip install -r requirements.txt
```

## Usage

### Run Basic Implementation:
```bash
cd BasicImplementation
python main.py
```

### Run PyTorch Implementation:
```bash
cd PyTorchImplementation
python main_pytorch.py
```

## Model Details

Both implementations predict drug usage based on:
1. Patient Count
2. Emergency Cases
3. Is Holiday (binary)
4. Previous Day Usage

**Algorithm**: Linear Regression with Gradient Descent
**Training**: 10,000 iterations
**Learning Rate**: 0.01
**Data Split**: 80% train, 20% test
