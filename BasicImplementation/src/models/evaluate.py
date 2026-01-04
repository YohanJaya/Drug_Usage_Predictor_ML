import numpy as np
from sklearn.metrics import mean_absolute_error


def quantile_loss(y_true, y_pred, q):
    """
    Pinball (quantile) loss
    """
    return np.mean(
        np.maximum(q * (y_true - y_pred),
                   (q - 1) * (y_true - y_pred))
    )


def evaluate_model(yTest, predictions, quantile):
    """
    Evaluate quantile regression properly.
    """

    # MAE only as a sanity check
    mae = mean_absolute_error(yTest, predictions)

    # Proper metric for quantile regression
    qloss = quantile_loss(yTest, predictions, quantile)

    coverage = (yTest <= predictions).mean()
    print(f"Coverage of {quantile}-quantile predictions: {coverage:.2%}")

    print("Model Evaluation:")
    print(f"MAE (sanity check): {mae:.2f}")
    print(f"Quantile Loss (q={quantile}): {qloss:.4f}")

    return {
        "mae": mae,
        "quantile_loss": qloss
    }
