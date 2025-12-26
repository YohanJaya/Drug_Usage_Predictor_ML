"""
Data generation and preprocessing modules
"""

from .synthetic_data_generator import generate_synthetic_data
from .preprocessing import normalize_features, split_data

__all__ = [
    "generate_synthetic_data",
    "normalize_features",
    "split_data",
]
