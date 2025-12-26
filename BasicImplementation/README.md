# Drug Usage Predictor - Basic NumPy Implementation

Linear Regression with manual Gradient Descent implementation using NumPy.

## Project Structure

```
BasicImplementation/
├── src/
│   ├── data/
│   │   ├── synthetic_data_generator.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── linear_regression.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   └── utils/
│       └── visualization.py
├── reports/figures/
├── main.py
└── README.md
```

## Features
- Manual implementation of gradient descent
- Linear regression from scratch
- Cost function calculation
- Gradient computation
- Multiple feature support
- Complete visualization suite

## Setup

Install dependencies (from parent folder):
```bash
pip install -r ../requirements.txt
```

## Usage

Run the complete pipeline:
```bash
python main.py
```

## Implementation Details

- **Gradient Descent**: Manually coded with cost and gradient functions
- **No ML frameworks**: Pure NumPy implementation
- **Educational**: See exactly how gradient descent works
- **Multiple features**: Supports any number of input features

## Key Functions

- `calCost()`: Calculate cost function J
- `calGradient()`: Calculate gradients for w and b
- `gradientDescent()`: Main training loop
- `train_model()`: High-level training interface
