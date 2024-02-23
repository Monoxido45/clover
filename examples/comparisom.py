import numpy as np
from sklearn.model_selection import train_test_split

from clover.cqr import ConformalizedQuantileRegression
from clover.locluster import KmeansSplit
from clover.locart import LocartSplit
from clover.models import QuantileGradientBoosting
from clover.scores import QuantileScore
from clover.simulation import simulation
from clover.utils import compute_interval_length, real_coverage

seed = 100
rng = np.random.RandomState(seed)

n_train = 2000
n_test_X = 5 * 10**3
n_test_y = 10**3

# Data
dim = 20
sim = simulation(dim=dim, coef=2, hetero_value=0.1)
sim.heteroscedastic(2 * n_train, random_seed=seed)

X_test = rng.uniform(-1.5, 1.5, size=(n_test_X, dim))
y_test = sim.heteroscedastic_r(X_test[:, 0], B=n_test_y)

X_train, X_calib, y_train, y_calib = train_test_split(
    sim.X, sim.y, test_size=0.5, random_state=seed
)

# Base model for CQR
base_model = QuantileGradientBoosting(alpha=0.05)

# Conformal methods
cqr = ConformalizedQuantileRegression(base_model, alpha=0.05)
locluster = KmeansSplit(
    QuantileScore, QuantileGradientBoosting, base_model_type=True, alpha=0.05
)
locart = LocartSplit(
    QuantileScore,
    QuantileGradientBoosting,
    alpha=0.05,
    base_mode_type=True,
    split_calib=False,
)

print("=> CQR")
cqr.fit(X_train, y_train)
cqr.calib(X_calib, y_calib)
cqr_pred = cqr.predict(X_test)
cqr_cov = real_coverage(cqr_pred, y_test)
cqr_len = compute_interval_length(cqr_pred)
print(np.mean(cqr_cov), np.mean(cqr_len))

print("=> Locluster")
locluster.fit(X_train, y_train)
locluster.calib(X_calib, y_calib)
locluster_pred = locluster.predict(X_test, length=2000)
locluster_cov = real_coverage(locluster_pred, y_test)
locluster_len = compute_interval_length(locluster_pred)
print(np.mean(locluster_cov), np.mean(locluster_len))

print("=> Locart")
locart.fit(X_train, y_train)
locart.calib(X_calib, y_calib)
locart_pred = locart.predict(X_test, length=2000)
locart_cov = real_coverage(locart_pred, y_test)
locart_len = compute_interval_length(locart_pred)
print(np.mean(locart_cov), np.mean(locart_len))
