# importing all needed packages
# importing mainly the mondrian regression split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from lcv.scores import LocalRegressionScore
from lcv.locart import MondrianRegressionSplit
from lcv.simulation import simulation
from lcv.utils import compute_interval_length, real_coverage, split
import numpy as np

# miscellanous
import os
from os import path
import time

# saving original path
original_path = os.getcwd()

def add_mondrian_conformal_statistics(kind = "homoscedastic",
           n_it = 200,
           n_train = np.array([500, 1000, 5000, 10000]),
           d = 20, 
           coef = 2,
           hetero_value = 0.25,
           asym_value = 0.6,
           t_degree = 4,
           random_seed = 1250,
           type_score = "regression",
           sig = 0.1,
           alpha = 0.5,
           B_x = 5000,
           B_y = 1000,
           base_model = RandomForestRegressor,
           **kwargs):
    # starting the experiment
    print("Starting experiments for {} data".format(kind))
    start_kind = time.time()
    # generating X's features from uniform
    X_test = np.random.uniform(low = -1.5, high = 1.5, size = (B_x, d))
    
    # testing if _V2 is inside kind
    # creating paths
    if "_V2" in kind:
      eta = 1.5
      kind = "asymmetric"
      folder_path = "/results/pickle_files/locart_experiments_results/{}_data_eta_{}".format(
        kind, eta)
    else:
      folder_path = "/results/pickle_files/locart_experiments_results/{}_data".format(
        kind)
        
    # creating directories to each file
    if not(path.exists(original_path + folder_path)):
      os.mkdir(original_path + folder_path)
        
    # simulating several y from same X
    sim_obj = simulation(dim = d, coef = coef, hetero_value = hetero_value, asym_value = asym_value, t_degree = t_degree)
    r_kind = getattr(sim_obj, kind + "_r")
    y_mat = r_kind(X_test[:, 0], B = B_y)
    
    # generating random_seed vector for the sake of reproducibility
    np.random.seed(random_seed)
    random_seeds = np.random.randint(0, 10**(8), n_it)
    
    for n in n_train:
      # testing wheter we already have all saved
      # if not, we run all and save all together in the same folder
      if not(path.exists(original_path + folder_path + "/{}_score_{}_dim_{}_{}_samples_measures/mondrian".format(
        kind, type_score, d, n))):
        print("running the mondrian experiments for {} training and calibration samples in the {} setting".format(n, kind))
        # measures to be saved at last
        mean_int_length_vector, median_int_length_vector = np.zeros(n_it), np.zeros(n_it)
        mean_diff_vector, median_diff_vector = np.zeros(n_it), np.zeros(n_it)
        mean_coverage_vector, median_coverage_vector = np.zeros(n_it), np.zeros(n_it)
        
        for i in range(n_it):
          if (i + 1) % 50 == 0:
            print("running {} iteration for {} training samples".format(i + 1, n))
          
          # simulating data and then splitting into train and calibration sets
          sim_kind = getattr(sim_obj, kind)
          sim_kind(2*n, random_seed = random_seeds[i])
          split_icp = split(sim_obj.X, sim_obj.y, test_size = 0.5, calibrate = False, random_seed = random_seeds[i])
          
          micp = MondrianRegressionSplit(base_model, alpha = sig, **kwargs)
          micp.fit(split_icp["X_train"], split_icp["y_train"])
          micp.calibrate(split_icp["X_test"], split_icp["y_test"])

          # mondrian icp real coverage and interval length
          micp_cond_r_real = real_coverage(micp.predict(X_test), y_mat)
          micp_interval_len = compute_interval_length(micp.predict(X_test))

          # computing micp measures
          mean_diff_vector[i], median_diff_vector[i] = np.mean(np.abs(micp_cond_r_real - (1 - sig))), np.median(np.abs(micp_cond_r_real - (1 - sig)))
          mean_coverage_vector[i], median_coverage_vector[i] = np.mean(micp_cond_r_real), np.median(micp_cond_r_real)
          mean_int_length_vector[i], median_int_length_vector[i] = np.mean(micp_interval_len), np.median(micp_interval_len)
          
        # creating directory
        os.mkdir(original_path + folder_path +"/{}_score_{}_dim_{}_{}_samples_measures/mondrian".format(
        kind, type_score, d, n))
        
        # changing working directory to the current folder
        os.chdir(original_path + folder_path +"/{}_score_{}_dim_{}_{}_samples_measures/mondrian".format(
        kind, type_score, d, n))
          
        # saving all matrices into npy files
        # interval length
        np.save("mean_interval_length_n_{}_{}_data.npy".format(
          n, kind), mean_int_length_vector)
        np.save("median_interval_length_n_{}_{}_data.npy".format(
          n, kind), median_int_length_vector)  
        
        # conditional difference
        np.save("mean_diff_n_{}_{}_data.npy".format(
          n, kind), mean_diff_vector)
        np.save("median_diff_n_{}_{}_data.npy".format(
          n, kind), median_diff_vector)
          
        # conditional difference
        np.save("mean_coverage_n_{}_{}_data.npy".format(
          n, kind), mean_coverage_vector)
        np.save("median_coverage_n_{}_{}_data.npy".format(
          n, kind), median_coverage_vector)
        
        # returning to original path
        os.chdir(original_path)
      
      else:
        continue
      
    print("Experiments finished for {} setting".format(kind))
    end_kind = time.time() - start_kind
    print("Time Elapsed to compute all metrics in the {} setting: {}".format(kind, end_kind))
    return end_kind
  
  

def compute_all_mondrian_conformal_statistics(
  kind_lists = ["homoscedastic", "heteroscedastic", "asymmetric", "asymmetric_V2", "t_residuals", "non_cor_heteroscedastic"],
  n_it = 200,
  n_train = np.array([500, 1000, 5000, 10000]),
  d = 20):
    print("Starting all experiments")
    start_exp = time.time()
    times_list = list()
    for kinds in kind_lists:
      times_list.append(add_mondrian_conformal_statistics(kind = kinds, n_it = n_it, n_train = n_train, d = d))
    end_exp = time.time() - start_exp
    print("Time elapsed to conduct all experiments: {}".format(end_exp))
    np.save("results/pickle_files/locart_experiments_results/mondrian_running_times.npy", np.array(times_list.append(end_exp)))
    return None

if __name__ == '__main__':
  print("We will now compute all conformal statistics for mondrian split for several simulated examples")
  compute_all_mondrian_conformal_statistics()
