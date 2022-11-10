import numpy as np
from sklearn.model_selection import train_test_split

from lcv.locluster import KmeansSplit
from lcv.models import QuantileGradientBoosting
from lcv.scores import QuantileScore
from lcv.simulation import simulation
from lcv.utils import compute_interval_length, real_coverage

n_train = 1000  # com 50 tbm
n_test_X = 5 * 10 ** 3
n_test_y = 10 ** 3
n_it = 5

random_seeds = np.random.randint(0, 10 ** (8), size=n_it)
rng = np.random.RandomState(1250)

dim = 20

for i in range(n_it):
    print(f"Running for seed {random_seeds[i]}")
    sim = simulation(dim=dim, coef=2)
    sim.homoscedastic(2 * n_train, random_seed=random_seeds[i])

    X_test = rng.uniform(-1.5, 1.5, size=(n_test_X, dim))
    y_test = sim.homoscedastic_r(X_test[:, 0], B=n_test_y)

    X_train, X_calib, y_train, y_calib = train_test_split(
        sim.X, sim.y, test_size=0.5, random_state=random_seeds[i]
    )

    locluster = KmeansSplit(QuantileScore, QuantileGradientBoosting, alpha=0.05)
    locluster.fit(X_train, y_train)
    locluster.calib(X_calib, y_calib)

    predictions = np.array(locluster.predict(X_test, length=2000))

    cov = real_coverage(predictions, y_test)
    interval_len = compute_interval_length(predictions)

    print(np.mean(cov), np.mean(interval_len))
