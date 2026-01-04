import numpy as np

def predict(model, xTest):
    """
    Make quantile predictions.
    """

    predictions = model.predict(xTest)

    # Demand / counts should not be negative
    predictions = np.maximum(predictions, 0)

    return predictions