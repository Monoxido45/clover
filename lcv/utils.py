import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.stats as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
    m = x.shape[0]  # sample size
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
        cover = np.logical_and(y_test >= predictions[:, 0], y_test <= predictions[:, 1])
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
        cover = np.logical_and(y_test >= predictions[:, 0], y_test <= predictions[:, 1])
        z = np.dot(X_test, v)
        idx = np.where((z >= a) * (z <= b))
        coverage = np.mean(cover[idx])
        return coverage

    X_train, X_test2, y_train, y_test2, pred_train, pred_test = train_test_split(
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
    coverage = wsc_vab(X_test2, y_test2, pred_test, v_star, a_star, b_star)
    return coverage


def wsc_coverage(
    X_test,
    y_test,
    predictions,
    delta=0.1,
    M=1000,
    random_state=2020,
    alpha=0.1,
    verbose=False,
):
    # worst slab coverage
    wsc_value, v, a, b = wsc(
        X_test, y_test, predictions, delta, M, random_state, verbose
    )

    return np.abs(wsc_value - (1 - alpha))


# proposing a mean of maximum and infimum slab coverage
def msc(
    X_test, y_test, predictions, delta=0.1, M=1000, random_state=1250, verbose=False
):
    rng = np.random.default_rng(random_state)

    def msc_v(X_test, y_test, predictions, delta, v):
        n = len(y_test)
        cover = np.logical_and(y_test >= predictions[:, 0], y_test <= predictions[:, 1])
        z = np.dot(X_test, v)

        # Compute mass
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0 - delta) * n))
        ai_best = 0
        bi_best = n
        cover_min = 1
        cover_max = 0
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai + int(np.round(delta * n)), n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1, n - ai + 1)
            min_coverage, max_coverage = np.copy(coverage), np.copy(coverage)
            min_coverage[np.arange(0, bi_min - ai)] = 1
            max_coverage[np.arange(0, bi_min - ai)] = 0

            bi_star = ai + np.argmin(min_coverage)
            cover_star = min_coverage[bi_star - ai]

            bi_star_max = ai + np.argmax(max_coverage)
            cover_star_max = max_coverage[bi_star_max - ai]

            if cover_star < cover_min:
                ai_best = ai
                bi_best = bi_star
                cover_min = cover_star

            if cover_star_max > cover_max:
                bi_best_max = bi_star_max
                cover_max = cover_star_max

        return (
            cover_max,
            cover_min,
            z_sorted[ai_best],
            z_sorted[bi_best],
            z_sorted[bi_best_max],
        )

    def sample_sphere(n, p):
        v = rng.normal(size=(p, n))
        v /= np.linalg.norm(v, axis=0)
        return v.T

    V = sample_sphere(M, p=X_test.shape[1])
    msc_min_list = [[]] * M
    msc_max_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    b_list_max = [[]] * M
    if verbose:
        for m in tqdm(range(M)):
            (
                msc_max_list[m],
                msc_min_list[m],
                a_list[m],
                b_list[m],
                b_list_max[m],
            ) = msc_v(X_test, y_test, predictions, delta, V[m])
    else:
        for m in range(M):
            (
                msc_max_list[m],
                msc_min_list[m],
                a_list[m],
                b_list[m],
                b_list_max[m],
            ) = msc_v(X_test, y_test, predictions, delta, V[m])

    idx_star = np.argmin(np.array(msc_min_list))
    idx_star_max = np.argmax(np.array(msc_max_list))
    a_star_min, a_star_max = a_list[idx_star], a_list[idx_star_max]
    b_star, b_star_max = b_list[idx_star], b_list[idx_star_max]
    v_min_star, v_max_star = V[idx_star], V[idx_star_max]
    msc_star, msc_star_max = msc_min_list[idx_star], msc_max_list[idx_star_max]

    return (
        msc_star,
        msc_star_max,
        a_star_min,
        a_star_max,
        b_star,
        b_star_max,
        v_min_star,
        v_max_star,
    )


def msc_unbiased(
    X_test,
    y_test,
    predictions,
    delta=0.1,
    M=1000,
    test_size=0.75,
    random_state=2020,
    verbose=False,
):
    def msc_vab(X_test, y_test, predictions, v_min, v_max, a_min, a_max, b_min, b_max):
        cover = np.logical_and(y_test >= predictions[:, 0], y_test <= predictions[:, 1])
        z_min = np.dot(X_test, v_min)
        z_max = np.dot(X_test, v_max)
        idx_min = np.where((z_min >= a_min) * (z_min <= b_min))
        idx_max = np.where((z_max >= a_max) * (z_max <= b_max))

        coverage_min = np.mean(cover[idx_min])
        coverage_max = np.mean(cover[idx_max])
        return (coverage_min + coverage_max) / 2

    X_train, X_test2, y_train, y_test2, pred_train, pred_test = train_test_split(
        X_test, y_test, predictions, test_size=test_size, random_state=random_state
    )
    # Find adversarial parameters
    (
        msc_star,
        msc_star_max,
        a_star_min,
        a_star_max,
        b_star,
        b_star_max,
        v_min_star,
        v_max_star,
    ) = msc(
        X_train,
        y_train,
        pred_train,
        delta=delta,
        M=M,
        random_state=random_state,
        verbose=verbose,
    )
    # Estimate coverage
    coverage = msc_vab(
        X_test2,
        y_test2,
        pred_test,
        v_min_star,
        v_max_star,
        a_star_min,
        a_star_max,
        b_star,
        b_star_max,
    )
    return coverage


def msc_coverage(
    X_test,
    y_test,
    predictions,
    delta=0.1,
    M=1000,
    random_state=2020,
    alpha=0.1,
    verbose=False,
):
    # worst slab coverage
    msc_value = msc_unbiased(
        X_test, y_test, predictions, delta, M, random_state, verbose
    )

    return np.abs(msc_value - (1 - alpha))


# proposing a new conditional coverage metric based on absolut difference for each slab
def wsd(
    X_test,
    y_test,
    predictions,
    alpha=0.1,
    delta=0.1,
    M=1000,
    random_state=2020,
    verbose=False,
):
    rng = np.random.default_rng(random_state)

    def wsd_v(X_test, y_test, predictions, delta, alpha, v):
        n = len(y_test)
        cover = np.logical_and(y_test >= predictions[:, 0], y_test <= predictions[:, 1])
        z = np.dot(X_test, v)

        # Compute mass
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0 - delta) * n))
        ai_best = 0
        bi_best = n
        cover_min = 1
        total_sample = 0
        sum = 0
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai + int(np.round(delta * n)), n)
            dif = np.abs(
                (np.cumsum(cover_ordered[ai:n]) / np.arange(1, n - ai + 1))
                - (1 - alpha)
            )
            new_dif = np.delete(dif, np.arange(0, bi_min - ai))
            total_sample += new_dif.shape[0]
            sum += np.sum(new_dif)
        return sum, total_sample

    def sample_sphere(n, p):
        v = rng.normal(size=(p, n))
        v /= np.linalg.norm(v, axis=0)
        return v.T

    V = sample_sphere(M, p=X_test.shape[1])
    wsd_val_list = [[]] * M
    wsd_tot_list = [[]] * M

    if verbose:
        for m in tqdm(range(M)):
            wsd_val_list[m], wsd_tot_list[m] = wsd_v(
                X_test, y_test, predictions, alpha, delta, V[m]
            )
    else:
        for m in range(M):
            wsd_val_list[m], wsd_tot_list[m] = wsd_v(
                X_test, y_test, predictions, alpha, delta, V[m]
            )

    return np.sum(np.array(wsd_val_list)) / np.sum(np.array(wsd_tot_list))


# our new conditional coverage measure given by a k-means coverage
def clustering_coverage(
    X_test,
    y_test,
    predictions,
    alpha=0.1,
    tune_k=True,
    prop_k=np.arange(2, 101, 1),
    random_seed=1250,
):
    # scaling features
    scaler = StandardScaler()
    new_X = scaler.fit_transform(X_test)
    current_sil = -1

    # fitting k-means and tuning k
    if tune_k and prop_k.shape[0] > 1:
        for k in prop_k:
            model = KMeans(n_clusters=k, random_state=random_seed, n_init=30).fit(new_X)
            labels = model.labels_
            new_sil = silhouette_score(new_X, labels, metric="euclidean")
            if new_sil > current_sil:
                current_model = model
                current_sil = new_sil
                current_k = k
                final_labels = labels
    else:
        current_model = KMeans(
            n_clusters=prop_k, random_state=random_seed, n_init=30
        ).fit(new_X)
        final_labels = current_model.labels_

    # separating predictions in each label
    dif = np.zeros(np.unique(final_labels).shape[0])
    i = 0
    for label in np.unique(final_labels):
        new_preds = predictions[np.where(final_labels == label)[0], :]
        new_y = y_test[np.where(final_labels == label)[0]]
        loc_cov = np.mean(
            np.logical_and(new_y >= new_preds[:, 0], new_y <= new_preds[:, 1])
        )
        dif[i] = np.abs(loc_cov - (1 - alpha))
        i += 1

    return np.mean(dif), dif


def clustering_CI_coverage(
    X_test,
    y_test,
    predictions,
    alpha=0.1,
    tune_k=True,
    prop_k=np.arange(5, 200, 5),
    min=100,
    random_seed=1250,
):
    # scaling features
    scaler = StandardScaler()
    new_X = scaler.fit_transform(X_test)
    tuned = False

    # fitting k-means and tuning k
    if tune_k and prop_k.shape[0] > 1:
        i = 0
        while tuned == False or i == (prop_k.shape[0] - 1):
            model = KMeans(
                n_clusters=prop_k[i], random_state=random_seed, n_init=30
            ).fit(new_X)
            labels = model.labels_
            min_number = np.min(np.unique(labels, return_counts=True)[1])

            if min_number > min:
                final_labels = labels
                final_k = prop_k[i]
            else:
                tuned = True
            i += 1

    else:
        current_model = KMeans(
            n_clusters=prop_k, random_state=random_seed, n_init=30
        ).fit(new_X)
        final_labels = current_model.labels_
        final_k = prop_k

    # separating predictions in each label
    id_fall = np.zeros(np.unique(final_labels).shape[0])
    i = 0
    for label in np.unique(final_labels):
        # constructing CI based on local coverage
        new_preds = predictions[np.where(final_labels == label)[0], :]
        new_y = y_test[np.where(final_labels == label)[0]]
        n = new_y.shape[0]
        loc_cov = np.mean(
            np.logical_and(new_y >= new_preds[:, 0], new_y <= new_preds[:, 1])
        )
        var_p = np.sqrt(loc_cov * (1 - loc_cov) / n)
        CI = np.array(
            [
                loc_cov - (st.norm.ppf(1 - (alpha / 2)) * var_p),
                loc_cov + (st.norm.ppf(1 - (alpha / 2)) * var_p),
            ]
        )

        # verifying if specified confidence level falls into confidence interval
        id_fall[i] = (CI[0] <= (1 - alpha) <= CI[1]) + 0
        i += 1

    return np.mean(id_fall), final_k


def ILS_coverage(predictions_1, predictions_2, y_test):
    L1 = compute_interval_length(predictions_1)
    L2 = compute_interval_length(predictions_2)
    delta_li = L1 - L2
    q_li = np.quantile(delta_li, q=0.9)
    ils_idx = np.where(delta_li >= q_li)[0]
    ils_cover_1 = np.mean(
        np.logical_and(
            y_test[ils_idx] >= predictions_1[ils_idx, 0],
            y_test[ils_idx] <= predictions_1[ils_idx, 1],
        )
        + 0
    )
    ils_cover_2 = np.mean(
        np.logical_and(
            y_test[ils_idx] >= predictions_2[ils_idx, 0],
            y_test[ils_idx] <= predictions_2[ils_idx, 1],
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

