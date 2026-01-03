"""
Linear regression model with gradient descent
"""

from .train import train_model
from .predict import predict
from .evaluate import evaluate_model

__all__ = [
    'train_model',
    'predict',
    'evaluate_model'

]
