import numpy as np

def generate_synthetic_data(days=365, random_state=None):
    """
    Generate synthetic data for drug usage prediction
    
    Args:
        days (int): Number of days of data to generate
        random_state (int): Random seed for reproducibility
        
    Returns:
        X (ndarray (m,n)): Feature matrix
        y (ndarray (m,)): Target variable (medicine usage)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    patient_count = np.random.randint(80, 350, days)
    emergency_cases = np.random.randint(5, 60, days)
    is_holiday = np.random.choice([0, 1], days, p=[0.85, 0.15])
    previous_day_usage = np.random.randint(300, 2000, days)

    # Feature matrix (m, n)
    X = np.c_[
        patient_count,
        emergency_cases,
        is_holiday,
        previous_day_usage
    ]

    # Target variable (medicine usage)
    y = (
        patient_count * 3 +
        emergency_cases * 12 -
        is_holiday * 200 +
        np.random.normal(0, 100, days)
    )

    y = y.astype(int)
    
    return X, y
