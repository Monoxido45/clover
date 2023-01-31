import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import tqdm


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
    return predictions[:, 1] - predictions[:, 0]


# metrics for conditional coverage for real data
# pearson correlation between coverage indicator and predictions
def pearson_correlation(predictions, y_test):
    # interval length
    L = compute_interval_length(predictions)
    V = np.logical_and(y_test >= predictions[:, 0], y_test <= predictions[:, 1]) + 0

    # computing pearson correlation between
    return np.corrcoef(L, V)[0, 1]


# HSIC correlation between coverage indicator and predictions considering the gaussian kernel
# code addapted from https://github.com/danielgreenfeld3/XIC/blob/master/hsic.py
def GaussianKernelMatrix(x, sigma=1):
    # x should be a 1d vector
    pairwise_distances_ = pairwise_distances(x.reshape(-1, 1))
    return np.exp(-((pairwise_distances_) ** 2) / (2 * sigma))


def HSIC(x, y, s_x=1, s_y=1):
    m, _ = x.shape  # sample size
    K = GaussianKernelMatrix(x, s_x)
    L = GaussianKernelMatrix(y, s_y)
    H = np.identity(m) - 1.0 / m * np.ones((m, m))
    HSIC = (np.trace(np.matmul(L, np.matmul(H, np.matmul(K, H))))) / ((m - 1) ** 2)

    return HSIC


def HSIC_correlation(predictions, y_test, s_L=1, s_V=1):
    # interval length
    L = compute_interval_length(predictions)
    V = np.logical_and(y_test >= predictions[:, 0], y_test <= predictions[:, 1]) + 0
    return HSIC(L, V, s_x=s_L, s_y=s_V)


# delta WSC - worst slab coverage minus marginal coverage
# code adapted from https://github.com/msesia/arc/blob/master/arc/coverage.py
def wsc(
    X_test, y_test, predictions, delta=0.1, M=1000, random_state=1250, verbose=False
):
    rng = np.random.default_rng(random_state)

    def wsc_v(X_test, y_test, predictions, delta, v):
        n = len(y_test)
        cover = (
            np.logical_and(y_test >= predictions[:, 0], y_test <= predictions[:, 1]) + 0
        )
        z = np.dot(X_test, v)

        # Compute mass
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0 - delta) * n))
        ai_best = 0
        bi_best = n
        cover_min = 1
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai + int(np.round(delta * n)), n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1, n - ai + 1)
            coverage[np.arange(0, bi_min - ai)] = 1
            bi_star = ai + np.argmin(coverage)
            cover_star = coverage[bi_star - ai]
            if cover_star < cover_min:
                ai_best = ai
                bi_best = bi_star
                cover_min = cover_star
        return cover_min, z_sorted[ai_best], z_sorted[bi_best]

    def sample_sphere(n, p):
        v = rng.normal(size=(p, n))
        v /= np.linalg.norm(v, axis=0)
        return v.T

    V = sample_sphere(M, p=X_test.shape[1])
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    if verbose:
        for m in tqdm(range(M)):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(
                X_test, y_test, predictions, delta, V[m]
            )
    else:
        for m in range(M):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(
                X_test, y_test, predictions, delta, V[m]
            )

    idx_star = np.argmin(np.array(wsc_list))
    a_star = a_list[idx_star]
    b_star = b_list[idx_star]
    v_star = V[idx_star]
    wsc_star = wsc_list[idx_star]
    return wsc_star, v_star, a_star, b_star


def wsc_unbiased(
    X_test,
    y_test,
    predictions,
    delta=0.1,
    M=1000,
    test_size=0.75,
    random_state=2020,
    verbose=False,
):
    def wsc_vab(X_test, y_test, predictions, v, a, b):
        cover = (
            np.logical_and(y_test >= predictions[:, 0], y_test <= predictions[:, 1]) + 0
        )
        z = np.dot(X_test, v)
        idx = np.where((z >= a) * (z <= b))
        coverage = np.mean(cover[idx])
        return coverage

    X_train, X_test, y_train, y_test, pred_train, pred_test = train_test_split(
        X_test, y_test, predictions, test_size=test_size, random_state=random_state
    )
    # Find adversarial parameters
    wsc_star, v_star, a_star, b_star = wsc(
        X_train,
        y_train,
        pred_train,
        delta=delta,
        M=M,
        random_state=random_state,
        verbose=verbose,
    )
    # Estimate coverage
    coverage = wsc_vab(X_test, y_test, pred_test, v_star, a_star, b_star)
    return coverage


def ILS_coverage(predictions_1, predictions_2, y_test):
    L1 = compute_interval_length(predictions_1)
    L2 = compute_interval_length(predictions_2)
    delta_li = L1 - L2
    q_li = np.quantile(delta_li, q=0.9)
    ils_idx = np.where(delta_li >= q_li)[0]
    ils_cover_1 = np.mean(
        np.logical_and(
            y_test[ils_idx] >= predictions_1[ils_idx, 0],
            y_test <= predictions_1[ils_idx, 1],
        )
        + 0
    )
    ils_cover_2 = np.mean(
        np.logical_and(
            y_test[ils_idx] >= predictions_2[ils_idx, 0],
            y_test <= predictions_2[ils_idx, 1],
        )
        + 0
    )
    marginal_cover_1 = np.mean(
        np.logical_and(y_test >= predictions_1[:, 0], y_test <= predictions_1[:, 1]) + 0
    )
    marginal_cover_2 = np.mean(
        np.logical_and(y_test >= predictions_2[:, 0], y_test <= predictions_2[:, 1]) + 0
    )
    delta_ils_1 = np.abs(ils_cover_1 - marginal_cover_1)
    delta_ils_2 = np.abs(ils_cover_2 - marginal_cover_2)
    return delta_ils_1 - delta_ils_2


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
