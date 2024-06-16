import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from nonconformist.cp import IcpRegressor
from nonconformist.nc import NcFactory
from tqdm import tqdm

from clover.locart import RegressionSplit
from clover.simulation import simulation

N_RUNS = 100
N_SAMPLES = 10000
RHO = 0.99
N_FEATURES = 20
N_RELEVANT_FEATURES = 5

seed_sequence = np.random.SeedSequence(entropy=0)
seeds = seed_sequence.generate_state(N_RUNS)

generator = simulation(
    dim=N_FEATURES, signif_vars=N_RELEVANT_FEATURES, rho=RHO, hetero_value=0.05
)

for i, seed in enumerate(tqdm(seeds)):
    generator.correlated_heteroscedastic(
        n=N_SAMPLES,
        random_seed=seed,
    )
    X, y = generator.X, generator.y

    X, X_test, y, y_test = train_test_split(X, y, test_size=3000)
    X, X_calib, y, y_calib = train_test_split(X, y, test_size=3000)

    model = RandomForestRegressor()

    # nonconformist version
    nc = NcFactory.create_nc(model)
    icp = IcpRegressor(nc)
    icp.fit(X, y)
    icp.calibrate(X_calib, y_calib)
    icp_pred = icp.predict(X_test, significance=0.1)

    # marginal coverage
    marg_cover = np.logical_and(y_test >= icp_pred[:, 0], y_test <= icp_pred[:, 1]) + 0
    tqdm.write(f"ICP: {marg_cover.mean(axis=0):3f}")

    # our version of reg-split
    reg_split = RegressionSplit(model, is_fitted=True, alpha=0.1)
    reg_split.fit(X, y)
    reg_split.calibrate(X_calib, y_calib)
    reg_split_pred = reg_split.predict(X_test)

    marg_cover = (
        np.logical_and(y_test >= reg_split_pred[:, 0], y_test <= reg_split_pred[:, 1]) + 0
    )
    tqdm.write(
        f"Reg-split: {marg_cover.mean(axis=0):3f}",
    )
