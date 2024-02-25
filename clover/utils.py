import numpy as np
from sklearn.model_selection import train_test_split

# several methods for computing conditional coverage and marginal coverage


# methods to compute coverage and interval length
# real coverage for simulated data
def real_coverage(model_preds, y_mat):
    r = np.zeros(model_preds.shape[0])
    for i in range(model_preds.shape[0]):
        r[i] = np.mean(
            np.logical_and(
                y_mat[i, :] >= model_preds[i, 0], y_mat[i, :] <= model_preds[i, 1]
            )
        )
    return r


# general interval length
def compute_interval_length(predictions):
    """
    Function to compute interval length for prediction intervals
    ----------------------------------------------------------------
    Input: (i)    predictions: Prediction interval vector.

    Output: Interval length vector.
    """
    return predictions[:, 1] - predictions[:, 0]


# implementing the standard interval score
def smis(predictions, y_test, alpha):
    """
    Function to standard interval score for prediction intervals at a miscalibration level alpha.
    ----------------------------------------------------------------
    Input: (i)    predictions: Prediction interval vector.
           (ii)   y_test: Testing label vector.
           (iii)  alpha: Miscalibration level used to build Prediction Intervals.

    Output: Smis score.
    """
    int_length = compute_interval_length(predictions)
    is_alpha = -(
        int_length
        + (2 / alpha * (predictions[:, 0] - y_test) * (y_test < predictions[:, 0]))
        + (2 / alpha * (y_test - predictions[:, 1]) * (y_test > predictions[:, 1]))
    )

    return np.mean(is_alpha)


# split function
def split(X, y, test_size=0.4, calib_size=0.5, calibrate=True, random_seed=1250):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    if calibrate:
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_train, y_train, test_size=calib_size, random_state=random_seed
        )
        return {
            "X_train": X_train,
            "X_calib": X_calib,
            "X_test": X_test,
            "y_train": y_train,
            "y_calib": y_calib,
            "y_test": y_test,
        }
    else:
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }
