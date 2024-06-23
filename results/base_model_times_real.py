# saving the time each base model runs for each setting
import numpy as np
import pandas as pd
import os
from os import path

# conformal methods
from nonconformist.cp import IcpRegressor
from nonconformist.nc import NcFactory

# base models
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from clover.utils import split

# simulation and performance measures
import time
from clover.simulation import simulation

original_path = os.getcwd()


def run_base_model_real(
    data_name,
    n_it=100,
    calib_size=0.5,
    test_size=0.2,
    base_model=RandomForestRegressor,
    random_seed=1250,
    **kwargs
):
    # starting experiment
    print("Starting experiments for {} data".format(data_name))

    # importing data into pandas data frame
    data_path = original_path + "/data/processed/" + data_name + ".csv"
    data = pd.read_csv(data_path)

    # separating y and X arrays
    y = data["target"].to_numpy()
    X = data.drop("target", axis=1).to_numpy()

    folder_path = "/results/pickle_files/real_data_experiments/{}_data".format(
        data_name
    )

    np.random.seed(random_seed)
    random_seeds = np.random.randint(0, 10 ** (8), n_it)
    var_path = "/{}_data_score_regression_model_time".format(data_name)

    if not (path.exists(original_path + folder_path + var_path)):
        running_time = np.zeros(n_it)
        print("running the experiments for {} data".format(data_name))
        for it in range(0, n_it):
            if (it + 1) % 25 == 0:
                print("Running iteration {}".format(it + 1))

            # splitting one part
            data = split(
                X,
                y,
                test_size=test_size,
                calib_size=calib_size,
                calibrate=True,
                random_seed=random_seeds[it],
            )

            start_loc = time.time()
            model = base_model(**kwargs)
            nc = NcFactory.create_nc(model)
            icp = IcpRegressor(nc)
            icp.fit(data["X_train"], data["y_train"])
            end_loc = time.time() - start_loc
            running_time[it] = end_loc

        # saving running times
        if not (path.exists(original_path + folder_path + var_path)):
            # creating directory
            os.mkdir(original_path + folder_path + var_path)

        # changing working directory to the current folder
        os.chdir(original_path + folder_path + var_path)

        # saving all matrices into npy files
        # interval length
        np.save("model_running_time_{}_data.npy".format(data_name), running_time)
        # returning to original path
        os.chdir(original_path)
    return None


if __name__ == "__main__":
    print("We will now compute all base model for each real data")
    model = input("Which model would like to use as base model? ")
    data = input("Which data would you like to select? ")
    n_it = int(input("How many iterations?"))
    if model == "Random Forest":
        random_state = 650
        run_base_model_real(data_name=data, n_it = n_it, random_state=random_state)
