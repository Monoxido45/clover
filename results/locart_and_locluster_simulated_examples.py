# importing all needed packages
# models being used
from lcv.locart import LocartSplit, LocalRegressionSplit
from lcv.locluster import KmeansSplit
from nonconformist.cp import IcpRegressor
from nonconformist.nc import NcFactory
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from lcv.scores import RegressionScore
import numpy as np

# miscellanous
import os
from os import path
import time

# importing simulation
original_path = os.getcwd()
os.chdir(original_path + "/results")
from simulation import simulation

# returning to original path
os.chdir(original_path)

# methods to compute coverage and interval length
def real_coverage(model_preds, y_mat):
    r = np.zeros(model_preds.shape[0])
    for i in range(model_preds.shape[0]):
        r[i] = np.mean(np.logical_and(y_mat[i,:] >= model_preds[i, 0], y_mat[i, :] <= model_preds[i, 1]))
    return r

def compute_interval_length(predictions):
    return(predictions[:, 1] - predictions[:, 0])
  
# split function
def split(X, y, test_size = 0.4, calibrate = True, random_seed = 1250):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = test_size,
                                                        random_state = random_seed)
    if calibrate:
        X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size = 0.3,
                                                             random_state = random_seed)
        return {"X_train":X_train, "X_calib": X_calib, "X_test" : X_test, 
                "y_train" : y_train, "y_calib" : y_calib, "y_test": y_test}
    else:
        return{"X_train":X_train,"X_test" : X_test, 
                "y_train" : y_train,"y_test": y_test}
  
  
def compute_conformal_statistics(kind = "homoscedastic",
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
           split_calib = True,
           n_estimators = 200,
           quantiles = [0.8, 0.85, 0.9, 0.95],
           random_states = [750, 85, 666, 69],
           prop_k = np.arange(2, 11),
           tune_k = True,
           criterion = "squared_error",
           max_depth = None,
           max_leaf_nodes = None,
           min_samples_leaf = 150,
           prune = True,
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
      if not(path.exists(original_path + folder_path + "/{}_score_{}_dim_{}_{}_samples_measures".format(
        kind, type_score, d, n))):
        print("running the experiments for {} training and calibration samples in the {} setting".format(n, kind))
        # measures to be saved at last
        mean_int_length_vector, median_int_length_vector = np.zeros((n_it, 5)), np.zeros((n_it, 5))
        mean_diff_vector, median_diff_vector = np.zeros((n_it, 5)), np.zeros((n_it, 5))
        mean_coverage_vector, median_coverage_vector = np.zeros((n_it, 5)), np.zeros((n_it, 5))
        
        for i in range(n_it):
          if (i + 1) % 50 == 0:
            print("running {} iteration for {} training samples".format(i + 1, n))
          
          # simulating data and then splitting into train and calibration sets
          sim_kind = getattr(sim_obj, kind)
          sim_kind(2*n, random_seed = random_seeds[i])
          split_icp = split(sim_obj.X, sim_obj.y, test_size = 0.5, calibrate = False, random_seed = random_seeds[i])
          
          # starting experiments, saving all tables in 
          if type_score == "regression":
              locluster_obj = KmeansSplit(nc_score = RegressionScore, base_model = base_model, alpha = sig, **kwargs)
              locluster_obj.fit(split_icp["X_train"], split_icp["y_train"])
              locluster_obj.calib(split_icp["X_test"], split_icp["y_test"], tune_k = tune_k, 
              prop_k = prop_k, n_estimators = n_estimators, quantiles = quantiles, random_states = random_states)
      
              # conditional coverage and interval length
              pred_locluster = np.array(locluster_obj.predict(X_test, length = 2000))
              cond_locluster_real =  real_coverage(pred_locluster, y_mat)
              locluster_interval_len = compute_interval_length(pred_locluster)
      
              # saving several measures
              mean_diff_vector[i, 0], median_diff_vector[i, 0] = np.mean(np.abs(cond_locluster_real - (1 - sig))), np.median(np.abs(cond_locluster_real - (1 - sig)))
              mean_coverage_vector[i, 0], median_coverage_vector[i, 0] = np.mean(cond_locluster_real), np.median(cond_locluster_real)
              mean_int_length_vector[i, 0], median_int_length_vector[i, 0] = np.mean(locluster_interval_len), np.median(locluster_interval_len)
      

              # fitting locart
              locart_obj = LocartSplit(nc_score = RegressionScore, base_model = base_model, alpha = sig, split_calib = split_calib, **kwargs)
              locart_obj.fit(split_icp["X_train"], split_icp["y_train"])
              locart_obj.calib(split_icp["X_test"], split_icp["y_test"], max_depth = max_depth, 
              max_leaf_nodes = max_leaf_nodes, min_samples_leaf = min_samples_leaf, criterion = criterion, prune_tree = prune)
              
              
              # conditional coverage and interval length
              pred_locart = np.array(locart_obj.predict(X_test, length = 2000))
              cond_locart_real =  real_coverage(pred_locart, y_mat)
              locart_interval_len = compute_interval_length(pred_locart)
      
              # several measures
              mean_diff_vector[i, 1], median_diff_vector[i, 1]  = np.mean(np.abs(cond_locart_real - (1 - sig))), np.median(np.abs(cond_locart_real - (1 - sig)))
              mean_coverage_vector[i, 1], median_coverage_vector[i, 1]  = np.mean(cond_locart_real), np.median(cond_locart_real)
              mean_int_length_vector[i, 1], median_int_length_vector[i, 1] = np.mean(locart_interval_len), np.median(locart_interval_len)
      
              # fitting default regression split
              model = base_model(**kwargs)
              nc = NcFactory.create_nc(model)
              icp = IcpRegressor(nc)
              icp.fit(split_icp["X_train"], split_icp["y_train"])
              icp.calibrate(split_icp["X_test"], split_icp["y_test"])
      
      
              # icp real coverage and interval length
              icp_cond_r_real = real_coverage(icp.predict(X_test, significance = sig), y_mat)
              icp_interval_len = compute_interval_length(icp.predict(X_test, significance = sig))
      
              # computing icp measures
              mean_diff_vector[i, 2], median_diff_vector[i, 2]  = np.mean(np.abs(icp_cond_r_real - (1 - sig))), np.median(np.abs(icp_cond_r_real - (1 - sig)))
              mean_coverage_vector[i, 2], median_coverage_vector[i, 2]  = np.mean(icp_cond_r_real), np.median(icp_cond_r_real)
              mean_int_length_vector[i, 2], median_int_length_vector[i, 2] = np.mean(icp_interval_len), np.median(icp_interval_len)
              
      
              # fitting wighted regression split
              wicp = LocalRegressionSplit(base_model, alpha = sig, **kwargs)
              wicp.fit(split_icp["X_train"], split_icp["y_train"])
              wicp.calibrate(split_icp["X_test"], split_icp["y_test"])

              # weighted icp real coverage and interval length
              wicp_cond_r_real = real_coverage(wicp.predict(X_test), y_mat)
              wicp_interval_len = compute_interval_length(wicp.predict(X_test))
      
              # computing wicp measures
              mean_diff_vector[i, 3], median_diff_vector[i, 3]  = np.mean(np.abs(wicp_cond_r_real - (1 - sig))), np.median(np.abs(wicp_cond_r_real - (1 - sig)))
              mean_coverage_vector[i, 3], median_coverage_vector[i, 3]  = np.mean(wicp_cond_r_real), np.median(wicp_cond_r_real)
              mean_int_length_vector[i, 3], median_int_length_vector[i, 3] = np.mean(wicp_interval_len), np.median(wicp_interval_len)
              

              # fitting uniform binning regression split
              locart_obj.uniform_binning(split_icp["X_test"], split_icp["y_test"])
      
              # computing local coverage to uniform binning
              pred_uniform = np.array(locart_obj.predict(X_test, length = 2000, type_model = "euclidean"))
              uniform_cond_r_real = real_coverage(pred_uniform, y_mat)
              uniform_interval_len = compute_interval_length(pred_uniform)
      
              # computing euclidean binning measures
              mean_diff_vector[i, 4], median_diff_vector[i, 4]  = np.mean(np.abs(uniform_cond_r_real - (1 - sig))), np.median(np.abs(uniform_cond_r_real- (1 - sig)))
              mean_coverage_vector[i, 4], median_coverage_vector[i, 4]  = np.mean(uniform_cond_r_real), np.median(uniform_cond_r_real)
              mean_int_length_vector[i, 4], median_int_length_vector[i, 4] = np.mean(uniform_interval_len), np.median(uniform_interval_len)

        # creating directory
        os.mkdir(original_path + folder_path +"/{}_score_{}_dim_{}_{}_samples_measures".format(
        kind, type_score, d, n))
        
        # changing working directory to the current folder
        os.chdir(original_path + folder_path +"/{}_score_{}_dim_{}_{}_samples_measures".format(
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

# method that make all the computations for all kinds of data

def compute_all_conformal_statistics(
  kind_lists = ["homoscedastic", "heteroscedastic", "asymmetric", "asymmetric_V2", "t_residuals", "non_cor_heteroscedastic"],
  n_it = 200,
  n_train = np.array([500, 1000, 5000, 10000]),
  d = 20):
    print("Starting all experiments")
    start_exp = time.time()
    times_list = list()
    for kinds in kind_lists:
      times_list.append(compute_conformal_statistics(kind = kinds, n_it = n_it, n_train = n_train, d = d))
    end_exp = time.time() - start_exp
    print("Time elapsed to conduct all experiments: {}".format(end_exp))
    np.save("results/pickle_files/locart_experiments_results/running_times.npy", np.array(times_list.append(end_exp)))
    return None

if __name__ == '__main__':
  print("We will now compute all conformal statistics for several simulated examples")
  compute_all_conformal_statistics()
      
      
    
    
