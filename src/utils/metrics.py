import numpy as np


def nwrmsle(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray = None
) -> float:
    """
    Normalized Weighted Root Mean Squared Logarithmic Error

    Parameters:
    -----------
    y_true : np.ndarray
        Actual values (may contain negative values for returns)
    y_pred : np.ndarray
        Predicted values
    weights : np.ndarray, optional
        Weights for each observation (from items.csv - perishable=1.25, else=1.0)

    Returns:
    --------
    float
        NWRMSLE score
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Clip predictions to avoid negative values (sales can't be negative)
    y_pred = np.maximum(0, y_pred)

    # CRITICAL FIX: For true values, we need to handle returns (negative sales)
    # log(1 + x) is undefined for x < -1, so we need to clip or transform
    # Option 1: Clip negative values to 0 (treat returns as no sale)
    # Option 2: Use log(1 + max(0, x)) - simpler and standard in competition
    y_true_processed = np.maximum(0, y_true)

    # Calculate log(1 + y) for both true and predicted
    log_true = np.log1p(y_true_processed)
    log_pred = np.log1p(y_pred)

    # Squared errors
    squared_errors = (log_pred - log_true) ** 2

    # Apply weights if provided
    if weights is not None:
        weights = np.asarray(weights)
        weighted_squared_errors = squared_errors * weights
        weighted_sum = np.sum(weighted_squared_errors)
        weight_sum = np.sum(weights)
    else:
        weighted_squared_errors = squared_errors
        weighted_sum = np.sum(weighted_squared_errors)
        weight_sum = len(y_true)

    # Calculate NWRMSLE
    result = np.sqrt(weighted_sum / weight_sum)

    # Check for inf/nan (shouldn't happen now)
    if np.isinf(result) or np.isnan(result):
        print(
            f"Warning: Invalid result. y_true min: {y_true.min()}, y_true max: {y_true.max()}"
        )
        print(f"y_pred min: {y_pred.min()}, y_pred max: {y_pred.max()}")
        print(f"Weighted sum: {weighted_sum}, weight sum: {weight_sum}")
        return 1.0  # fallback

    return float(result)
