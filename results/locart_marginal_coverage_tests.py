# test splitting coverage
import numpy as np
import pandas as pd
import os
from os import path

# base models and graph tools
from sklearn.ensemble import RandomForestRegressor

# conformal methods
from clover.locart import LocartSplit
from clover.scores import RegressionScore

# importing LCP-RF
from acpi import ACPI

# performance measures
import time
from clover.utils import compute_interval_length, split
import gc

# original path
original_path = os.getcwd()


def compute_metrics(
    data_name,
    n_it=100,
    base_model=RandomForestRegressor,
    sig=0.1,
    test_size=0.2,
    is_fitted=True,
    completing=False,
    save_all=True,
    calib_size=0.5,
    random_seed=1250,
    random_projections=False,
    h=20,
    m=300,
    split_calib=True,
    criterion="squared_error",
    max_depth=None,
    max_leaf_nodes=None,
    min_samples_leaf=150,
    prune=True,
    **kwargs
):
    # starting experiment
    print("Starting  locart tests for {} data".format(data_name))
    start_kind = time.time()

    # importing data into pandas data frame
    data_path = original_path + "/data/processed/" + data_name + ".csv"
    data = pd.read_csv(data_path)

    # separating y and X arrays
    y = data["target"].to_numpy()
    X = data.drop("target", axis=1).to_numpy()

    print(
        "Number of samples that will be used for training and calibration: {}".format(
            (1 - test_size) * X.shape[0]
        )
    )
    print("Number of samples used for testing: {}".format(test_size * X.shape[0]))

    # managing directories
    folder_path = "/results/pickle_files/locart_tests/{}_data".format(data_name)

    # creating directories to each file
    if not (path.exists(original_path + folder_path)):
        os.mkdir(original_path + folder_path)

    # generating two random seeds vector
    np.random.seed(random_seed)
    random_seeds = np.random.randint(0, 10 ** (8), n_it)

    # testing wheter we already have all saved
    # if not, we run all and save all together in the same folder
    var_path = "/{}_data_score_regression_measures".format(data_name)
    if not (path.exists(original_path + folder_path + var_path)) or completing:
        if not completing:
            print("running locart tests for {} data".format(data_name))
            # measures to be saved at last
            mean_int_length_vector, mean_coverage_vector = np.zeros(n_it), np.zeros(
                n_it
            )
            init_it = 0
        else:
            os.chdir(original_path + folder_path + var_path)
            print("continuing locart tests for {} data".format(data_name))
            # measures to be saved at last
            # real measures
            mean_int_length_vector = np.load(
                "mean_interval_length_{}_data.npy".format(data_name)
            )

            mean_coverage_vector = np.load(
                "mean_coverage_{}_data.npy".format(data_name)
            )

            init_it = iter_completing

        for it in range(init_it, n_it):
            if (it + 1) % 25 == 0:
                print("running {} iteration for {} data".format(it + 1, data_name))
            seed = random_seeds[it]

            # splitting data into train, test and calibration sets
            data = split(
                X,
                y,
                test_size=test_size,
                calib_size=calib_size,
                calibrate=True,
                random_seed=seed,
            )
            X_test, y_test = data["X_test"], data["y_test"]

            # fitting base model
            model = base_model(**kwargs).fit(data["X_train"], data["y_train"])

            # fitting all methods, saving running times and each metric
            # fitting normal locart
            start_loc = time.time()
            locart_obj = LocartSplit(
                nc_score=RegressionScore,
                cart_type="CART",
                base_model=model,
                is_fitted=is_fitted,
                alpha=sig,
                split_calib=split_calib,
                **kwargs
            )
            locart_obj.fit(data["X_train"], data["y_train"])
            locart_obj.calib(
                data["X_calib"],
                data["y_calib"],
                max_depth=max_depth,
                max_leaf_nodes=max_leaf_nodes,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                prune_tree=prune,
                random_projections=random_projections,
                m=m,
                h=h,
            )

            end_loc = time.time() - start_loc

            # predictions
            locart_pred = np.array(locart_obj.predict(X_test))

            # marginal coverage
            marg_cover = (
                np.logical_and(y_test >= locart_pred[:, 0], y_test <= locart_pred[:, 1])
                + 0
            )

            mean_coverage_vector[it] = np.mean(marg_cover)

            # mean interval length
            mean_int_length_vector[it] = np.mean(compute_interval_length(locart_pred))

            # deletting objects and removing from memory
            del locart_obj
            del locart_pred
            gc.collect()

            if (it + 1) % 25 == 0:
                print("Saving data checkpoint on iteration {}".format(it + 1))

            if (it + 1) % 25 == 0 or (it + 1 == 1) or save_all:
                # saving checkpoint of metrics
                saving_metrics(
                    original_path,
                    folder_path,
                    var_path,
                    data_name,
                    mean_int_length_vector,
                    mean_coverage_vector,
                )

            # saving all metrics again
            saving_metrics(
                original_path,
                folder_path,
                var_path,
                data_name,
                mean_int_length_vector,
                mean_coverage_vector,
            )

    print("Locart testing finished for {} data".format(data_name))
    end_kind = time.time() - start_kind
    print(
        "Time Elapsed to compute metrics in the {} data: {}".format(data_name, end_kind)
    )
    return end_kind


def saving_metrics(
    original_path,
    folder_path,
    var_path,
    data_name,
    mean_int_length_vector,
    mean_coverage_vector,
):
    # checking if path exsist
    if not (path.exists(original_path + folder_path + var_path)):
        # creating directory
        os.mkdir(original_path + folder_path + var_path)

    # changing working directory to the current folder
    os.chdir(original_path + folder_path + var_path)

    # saving all matrices into npy files
    # interval length
    np.save(
        "mean_interval_length_{}_data.npy".format(data_name), mean_int_length_vector
    )

    # mean coverage
    np.save("mean_coverage_{}_data.npy".format(data_name), mean_coverage_vector)

    # returning to original path
    os.chdir(original_path)


data_names = [
    "winewhite",
    "winered",
    "concrete",
    "airfoil",
    "electric",
    "superconductivity",
    "cycle",
    "protein",
    "news",
    "bike",
    "star",
    "meps19",
]


if __name__ == "__main__":
    print("We will now compute all conformal statistics for real data")
    model = input("Which model would like to use as base model? ")
    if model == "Random Forest":
        random_state = 650

        for data in data_names:
            exp_time = compute_metrics(
                data_name=data,
                base_model=RandomForestRegressor,
                random_state=random_state,
            )
            print("Time elapsed to conduct testing: {}".format(exp_time))

            np.save(
                "results/pickle_files/locart_tests/{}_running_time.npy".format(data),
                np.array([exp_time]),
            )

        folder_path = "/results/pickle_files/locart_tests/{}_data/{}_data_score_regression_measures".format(
            data, data
        )

    # if data is already built, just obtain mean coverage for each data
    marginal_covers, ses = [], []
    for data in data_names:
        folder_path = "/results/pickle_files/locart_tests/{}_data/{}_data_score_regression_measures".format(
            data, data
        )

        marginal_coverage = np.load(
            original_path + folder_path + "/mean_coverage_{}_data.npy".format(data)
        )
        marginal_covers.append(np.mean(marginal_coverage))
        ses.append(2 * np.std(marginal_coverage))

    print(
        pd.DataFrame(
            {"data": data_names, "marginal_cover": marginal_covers, "2*se": ses}
        )
    )
