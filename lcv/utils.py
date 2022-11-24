import numpy as np
from sklearn.model_selection import train_test_split


# methods to compute coverage and interval length
def real_coverage(model_preds, y_mat):
    r = np.zeros(model_preds.shape[0])
    for i in range(model_preds.shape[0]):
        r[i] = np.mean(
            np.logical_and(
                y_mat[i, :] >= model_preds[i, 0], y_mat[i, :] <= model_preds[i, 1]
            )
        )
    return r


def compute_interval_length(predictions):
    return predictions[:, 1] - predictions[:, 0]


# split function
def split(X, y, test_size=0.4, calibrate=True, random_seed=1250):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    if calibrate:
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_train, y_train, test_size=0.3, random_state=random_seed
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
