import numpy as np
import pandas as pd
import os
from os import path

# base models and graph tools
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# conformal methods
from nonconformist.cp import IcpRegressor
from nonconformist.nc import NcFactory
from lcv.locart import LocalRegressionSplit, LocartSplit, MondrianRegressionSplit, QuantileSplit
from lcv.locart import LocartSplit, MondrianRegressionSplit
from lcv.locluster import KmeansSplit
from lcv.scores import RegressionScore, LocalRegressionScore

# importing LCP-RF
from acpi import ACPI

# simulation and performance measures
import time
from lcv.simulation import simulation
from lcv.utils import compute_interval_length, split, real_coverage, smis, wsc_coverage, pearson_correlation, HSIC_correlation
from lcv.valid_pred_sets import Valid_pred_sets

original_path = os.getcwd()

def compute_metrics_sim(
  n_it = 100,
  n_train = 10000,
  completing = False,
  iter_completing = 50,
  kind = "homoscedastic",
  p = np.array([1, 3, 5]),
  d = 20,
  hetero_value = 0.25,
  asym_value = 0.6,
  t_degree = 4,
  base_model = RandomForestRegressor, 
  sig = 0.1,
  valid_test_size = 0.2,
  valid_split = False,
  valid_min_sample = 150,
  save_all = True,
  valid_prune = True,
  calib_size = 0.5, 
  coef = 2,
  B_x = 5000,
  B_y = 1000, 
  random_seed = 1250,
  random_projections = False,
  h = 20,
  m = 300,
  split_calib = False,
  mad_model_cte = False,
  nbins = 30,
  criterion = "squared_error",
  max_depth = None,
  max_leaf_nodes = None,
  min_samples_leaf = 150,
  prune = True,
  **kwargs):
    # starting experiment
    print("Starting experiments for {} data with p = {}".format(kind, p))
    start_kind = time.time()
    
    # managing directories
    # testing if _V2 is inside kind
    # creating paths
    if "_V2" in kind:
      asym_value = 1.5
      kind = "asymmetric"
      # folder_path = "/results/pickle_files/locart_all_metrics_experiments/{}_data_eta_{}".format(
      #   kind, asym_value)
    else:
      folder_path = "/results/pickle_files/locart_all_metrics_experiments/{}_data".format(
        kind)
        
    # creating directories to each file
    if not(path.exists(original_path + folder_path)):
      os.mkdir(original_path + folder_path)
    
    # generating two random seeds vector
    np.random.seed(random_seed)
    random_seeds = np.random.randint(0, 10**(8), n_it)
    random_seeds_X = np.random.randint(0, 10**(8), n_it)
    
    for n_var in p:
      # testing wheter we already have all saved
      # if not, we run all and save all together in the same folder
      var_path = "/{}_score_regression_p_{}_{}_samples_measures".format(kind, n_var, n_train)
      if not(path.exists(original_path + folder_path + var_path)) or completing:
        if not completing:
          print("running the experiments for {} significant variables and {} training instances in the {} setting".format(n_var, n_train, kind))
          # measures to be saved at last
          # real measures
          mean_int_length_vector = np.zeros((n_it, 9))
          mean_diff_vector, median_diff_vector, max_diff_vector = np.zeros((n_it, 9)), np.zeros((n_it, 9)), np.zeros((n_it, 9))
          mean_coverage_vector = np.zeros((n_it, 9))
        
          # estimated measures
          smis_vector, wsc_vector = np.zeros((n_it, 9)), np.zeros((n_it, 9))
          pcor_vector, HSIC_vector = np.zeros((n_it, 9)), np.zeros((n_it, 9)) 
          mean_valid_pred_set, max_valid_pred_set = np.zeros((n_it, 9)), np.zeros((n_it, 9))
        
          # running times
          times = np.zeros((n_it, 9))
          init_it = 0
        else:
          os.chdir(original_path + folder_path + var_path)
          print("continuing experiments for {} significant variables and {} training instances in the {} setting".format(
            n_var, n_train, kind))
          # measures to be saved at last
          # real measures
          mean_int_length_vector = np.load("mean_interval_length_p_{}_{}_data.npy".format(
          n_var, kind))
          mean_diff_vector, median_diff_vector, max_diff_vector = (np.load("mean_diff_p_{}_{}_data.npy".format(
          n_var, kind)), np.load("median_diff_p_{}_{}_data.npy".format(
          n_var, kind)), np.load("max_diff_p_{}_{}_data.npy".format(
          n_var, kind)))
          mean_coverage_vector = np.load("mean_coverage_p_{}_{}_data.npy".format(
          n_var, kind))
        
          # estimated measures
          smis_vector, wsc_vector =  (np.load("smis_p_{}_{}_data.npy".format(
          n_var, kind)), np.load("wsc_p_{}_{}_data.npy".format(
          n_var, kind)))
        
          # running times
          times = np.load("run_times_p_{}_{}_data.npy".format(
            n_var, kind))
          init_it = iter_completing
          
        noise = (n_var == 1)
        for it in range(init_it, n_it):
          if (it + 1) % 25 == 0:
            print("running {} iteration for {} significant variables".format(it + 1, n_var))
          
          seed_X = random_seeds_X[it]
          seed = random_seeds[it]
          
          # type of simulation
          sim_obj = simulation(dim = d, coef = coef, hetero_value = hetero_value, 
          noise = noise, signif_vars = n_var, asym_value = asym_value, t_degree = t_degree)
          r_kind = getattr(sim_obj, kind + "_r")
          sim_kind = getattr(sim_obj, kind)
          
          # generating testing samples
          np.random.seed(seed_X)
          X_test = np.random.uniform(low = -1.5, high = 1.5, size = (B_x, d))
          if noise:
            X_grid = X_test[:, 0]
          else:
            X_grid = X_test
          
          # generating y_test
          y_test = r_kind(X_grid, B = 1).flatten() 
          
          # simulating train and calibration sets
          sim_kind(2*n_train, random_seed = random_seed)
          data = split(sim_obj.X, sim_obj.y, test_size = calib_size, calibrate = False, 
          random_seed = seed)
          
          # matrix of y's associated to each X in test set
          if noise:
            y_mat = r_kind(X_test[:, 0], B = B_y)
          else:
            y_mat = r_kind(X_test, B = B_y)
          
          # fitting all methods, saving running times and each metric
          # fitting normal locart
          start_loc = time.time()
          locart_obj = LocartSplit(nc_score = RegressionScore, cart_type = "CART", 
          base_model = base_model, alpha = sig, split_calib = split_calib, **kwargs)
          locart_obj.fit(data["X_train"], data["y_train"])
          locart_obj.calib(data["X_test"], data["y_test"], max_depth = max_depth, 
          max_leaf_nodes = max_leaf_nodes, min_samples_leaf = min_samples_leaf, criterion = criterion, prune_tree = prune, 
          random_projections = random_projections, m = m, h = h)
          
          end_loc = time.time() - start_loc
          times[it, 0] = end_loc
          
          # predictions
          locart_pred = np.array(locart_obj.predict(X_test))
          cond_locart_real = real_coverage(locart_pred, y_mat)
      
          # average, median and max distance
          dif_locart = np.abs(cond_locart_real - (1 - sig))
          mean_diff_vector[it, 0], median_diff_vector[it, 0], max_diff_vector[it, 0] = (np.mean(dif_locart), 
          np.median(dif_locart), np.max(dif_locart))
      
      
          # valid pred sets
          locart_valid = Valid_pred_sets(conf = locart_obj, alpha = sig, coverage_evaluator = "CART", prune = valid_prune, 
          split_train = valid_split)
          locart_valid.fit(X_test, y_test, test_size = valid_test_size, min_samples_leaf = valid_min_sample)
          mean_valid_pred_set[it, 0], max_valid_pred_set[it, 0] = locart_valid.compute_dif()
          
          # marginal coverage
          marg_cover = np.logical_and(y_test >= locart_pred[:, 0], 
              y_test <= locart_pred[:, 1]) + 0
          mean_coverage_vector[it, 0] = np.mean(
              marg_cover
          )
      
          # smis
          smis_vector[it, 0] = smis(locart_pred, y_test, alpha = sig)
      
          # mean interval length
          mean_int_length_vector[it, 0] = np.mean(compute_interval_length(locart_pred))
          
          # wsc
          wsc_vector[it, 0] = wsc_coverage(X_test, y_test, locart_pred)
          
          # correlations
          pcor_vector[it, 0] = pearson_correlation(locart_pred, y_test)
          HSIC_vector[it, 0] = HSIC_correlation(locart_pred, y_test)
      
          # fitting normal RF-locart
          start_loc = time.time()
          rf_locart_obj = LocartSplit(nc_score = RegressionScore, cart_type = "RF", base_model = base_model, 
          alpha = sig, split_calib = split_calib, **kwargs)
          rf_locart_obj.fit(data["X_train"], data["y_train"])
          rf_locart_obj.calib(data["X_test"], data["y_test"], max_depth = max_depth, 
          max_leaf_nodes = max_leaf_nodes, min_samples_leaf = min_samples_leaf, criterion = criterion, 
          prune_tree = prune, random_projections = random_projections, 
          m = m, h = h)
          
          end_loc = time.time() - start_loc
          times[it, 1] = end_loc
          
          # predictions
          rf_locart_pred = np.array(rf_locart_obj.predict(X_test))
          cond_rf_locart_real = real_coverage(rf_locart_pred, y_mat)
      
          # average, median and max distance
          dif_rf_locart = np.abs(cond_rf_locart_real - (1 - sig))
          mean_diff_vector[it, 1], median_diff_vector[it, 1], max_diff_vector[it, 1] = (np.mean(dif_rf_locart), 
          np.median(dif_rf_locart), np.max(dif_rf_locart))
      
      
          # valid pred sets
          rf_locart_valid = Valid_pred_sets(conf = rf_locart_obj, alpha = sig, coverage_evaluator = "CART", 
          prune = valid_prune, split_train = valid_split)
          rf_locart_valid.fit(X_test, y_test, test_size = valid_test_size, min_samples_leaf = valid_min_sample)
          mean_valid_pred_set[it, 1], max_valid_pred_set[it, 1] = rf_locart_valid.compute_dif()
      
          # smis
          smis_vector[it, 1] = smis(rf_locart_pred, y_test, alpha = sig)
          
          # wsc
          wsc_vector[it, 1] = wsc_coverage(X_test, y_test, rf_locart_pred)
          
          pcor_vector[it, 1] = pearson_correlation(rf_locart_pred, y_test)
          HSIC_vector[it, 1] = HSIC_correlation(rf_locart_pred, y_test)
      
          # mean interval length
          mean_int_length_vector[it, 1] = np.mean(compute_interval_length(rf_locart_pred))
      
          # marginal coverage
          marg_cover = np.logical_and(y_test >= rf_locart_pred[:, 0], 
              y_test <= rf_locart_pred[:, 1]) + 0
          mean_coverage_vector[it, 1] = np.mean(
              marg_cover
          )
      
      
          # fitting normal difficulty locart
          start_loc = time.time()
          dlocart_obj = LocartSplit(nc_score = RegressionScore, cart_type = "CART", 
          base_model = base_model, alpha = sig, split_calib = split_calib, weighting = True,**kwargs)
          dlocart_obj.fit(data["X_train"], data["y_train"])
          dlocart_obj.calib(data["X_test"], data["y_test"], max_depth = max_depth, 
          max_leaf_nodes = max_leaf_nodes, min_samples_leaf = min_samples_leaf, criterion = criterion, 
          prune_tree = prune, random_projections = random_projections, 
          m = m, h = h)
          
          end_loc = time.time() - start_loc
          times[it, 2] = end_loc
      
          # predictions
          dlocart_pred = np.array(dlocart_obj.predict(X_test))
          cond_dlocart_real = real_coverage(dlocart_pred, y_mat)
      
          # average, median and max distance
          dif_dlocart = np.abs(cond_dlocart_real - (1 - sig))
          mean_diff_vector[it, 2], median_diff_vector[it, 2], max_diff_vector[it, 2] = (np.mean(dif_dlocart), 
          np.median(dif_dlocart), np.max(dif_dlocart))
      
      
          # valid pred sets
          dlocart_valid = Valid_pred_sets(conf = dlocart_obj, alpha = sig, coverage_evaluator = "CART", 
          prune = valid_prune, split_train = valid_split)
          dlocart_valid.fit(X_test, y_test, test_size = valid_test_size, min_samples_leaf = valid_min_sample)
          mean_valid_pred_set[it, 2], max_valid_pred_set[it, 2] = dlocart_valid.compute_dif()
      
          # smis
          smis_vector[it, 2] = smis(dlocart_pred, y_test, alpha = sig)
          
          # wsc
          wsc_vector[it, 2] = wsc_coverage(X_test, y_test, dlocart_pred)
          
          pcor_vector[it, 2] = pearson_correlation(dlocart_pred, y_test)
          HSIC_vector[it, 2] = HSIC_correlation(dlocart_pred, y_test)
      
          # mean interval length
          mean_int_length_vector[it, 2]  = np.mean(compute_interval_length(dlocart_pred))
      
          # marginal coverage
          marg_cover = np.logical_and(y_test >= dlocart_pred[:, 0], 
              y_test <= dlocart_pred[:, 1]) + 0
          mean_coverage_vector[it, 2] = np.mean(
              marg_cover
          )
      
      
          # fitting RF difficulty locart
          start_loc = time.time()
          rf_dlocart_obj = LocartSplit(nc_score = RegressionScore, cart_type = "RF", base_model = base_model, 
          alpha = sig, split_calib = split_calib, weighting = True, **kwargs)
          rf_dlocart_obj.fit(data["X_train"], data["y_train"])
          rf_dlocart_obj.calib(data["X_test"], data["y_test"], max_depth = max_depth, 
          max_leaf_nodes = max_leaf_nodes, min_samples_leaf = min_samples_leaf, criterion = criterion, 
          prune_tree = prune, random_projections = random_projections, 
          m = m, h = h)
          
          end_loc = time.time() - start_loc
          times[it, 3] = end_loc
      
          # predictions
          rf_dlocart_pred = np.array(rf_dlocart_obj.predict(X_test))
          cond_rf_dlocart_real = real_coverage(rf_dlocart_pred, y_mat)
      
          # average, median and max distance
          dif_rf_dlocart = np.abs(cond_rf_dlocart_real - (1 - sig))
          mean_diff_vector[it, 3], median_diff_vector[it, 3], max_diff_vector[it, 3] = (np.mean(dif_rf_dlocart), 
          np.median(dif_rf_dlocart), np.max(dif_rf_dlocart))
      
      
          # valid pred sets
          rf_dlocart_valid = Valid_pred_sets(conf = rf_dlocart_obj, alpha = sig, coverage_evaluator = "CART", 
          prune = valid_prune, split_train = valid_split)
          rf_dlocart_valid.fit(X_test, y_test, test_size = valid_test_size, min_samples_leaf = valid_min_sample)
          mean_valid_pred_set[it, 3], max_valid_pred_set[it, 3] = rf_dlocart_valid.compute_dif()
      
          # smis
          smis_vector[it, 3] = smis(rf_dlocart_pred, y_test, alpha = sig)
      
          # mean interval length
          mean_int_length_vector[it, 3] = np.mean(compute_interval_length(rf_dlocart_pred))
          
          # wsc
          wsc_vector[it, 3] = wsc_coverage(X_test, y_test, rf_dlocart_pred)
          
          pcor_vector[it, 3] = pearson_correlation(rf_dlocart_pred, y_test)
          HSIC_vector[it, 3] = HSIC_correlation(rf_dlocart_pred, y_test)
      
          # marginal coverage
          marg_cover = np.logical_and(y_test >= rf_dlocart_pred[:, 0], 
              y_test <= rf_dlocart_pred[:, 1]) + 0
          mean_coverage_vector[it, 3] = np.mean(
              marg_cover
          )

      
          # fitting ACPI/LCP-RF
          start_loc = time.time()
      
          model = base_model(**kwargs)
          model.fit(data["X_train"], data["y_train"])
          acpi = ACPI(model_cali = model, n_estimators = 100)
          acpi.fit(data["X_test"], data["y_test"], nonconformity_func = None)
          acpi.fit_calibration(data["X_test"], data["y_test"], quantile = 1 - sig, only_qrf = True)
      
          end_loc = time.time() - start_loc
          times[it, 4] = end_loc
          
          acpi_pred = np.stack((acpi.predict_pi(X_test, method = "qrf")), axis = -1)
          cond_acpi_real = real_coverage(acpi_pred, y_mat)
      
          # valid pred sets
          acpi_valid = Valid_pred_sets(conf = acpi, alpha = sig, islcp = True, coverage_evaluator = "CART", 
          prune = valid_prune, split_train = valid_split)
          acpi_valid.fit(X_test, y_test, test_size = valid_test_size, min_samples_leaf = valid_min_sample)
          mean_valid_pred_set[it, 4], max_valid_pred_set[it, 4] = acpi_valid.compute_dif()
      
          # average, median and max distance
          dif_acpi = np.abs(cond_acpi_real - (1 - sig))
          mean_diff_vector[it, 4], median_diff_vector[it, 4], max_diff_vector[it, 4] = (np.mean(dif_acpi), 
          np.median(dif_acpi), np.max(dif_acpi))
      
          # smis
          smis_vector[it, 4] = smis(acpi_pred, y_test, alpha = sig)
          
          # wsc
          wsc_vector[it, 4] = wsc_coverage(X_test, y_test, acpi_pred)
          
          pcor_vector[it, 4] = pearson_correlation(acpi_pred, y_test)
          HSIC_vector[it, 4] = HSIC_correlation(acpi_pred, y_test)
      
          # mean interval length
          mean_int_length_vector[it, 4] = np.mean(compute_interval_length(acpi_pred))
      
          # marginal coverage
          marg_cover = np.logical_and(y_test >= acpi_pred[:, 0], 
              y_test <= acpi_pred[:, 1]) + 0
          mean_coverage_vector[it, 4] = np.mean(
              marg_cover
          )
          
      
          # fitting wlocart
          start_loc = time.time()
      
          wlocart_obj = LocartSplit(nc_score = LocalRegressionScore, cart_type = "RF", base_model = base_model, 
          alpha = sig, split_calib = split_calib, **kwargs)
          wlocart_obj.fit(data["X_train"], data["y_train"], mad_model_cte = mad_model_cte)
          wlocart_obj.calib(data["X_test"], data["y_test"], max_depth = max_depth, 
              max_leaf_nodes = max_leaf_nodes, min_samples_leaf = min_samples_leaf, criterion = criterion, prune_tree = prune,
              random_projections = random_projections, m = m, h = h)
          
          end_loc = time.time() - start_loc
          times[it, 5] = end_loc
      
          # predictions
          wlocart_pred = np.array(wlocart_obj.predict(X_test))
          cond_wlocart_real = real_coverage(wlocart_pred, y_mat)
      
          # average, median and max distance
          dif_wlocart = np.abs(cond_wlocart_real - (1 - sig))
          mean_diff_vector[it, 5], median_diff_vector[it, 5], max_diff_vector[it, 5] = (np.mean(dif_wlocart), 
          np.median(dif_wlocart), np.max(dif_wlocart))
      
      
          # valid pred sets
          wlocart_valid = Valid_pred_sets(conf = wlocart_obj, alpha = sig, coverage_evaluator = "CART", 
          prune = valid_prune, split_train = valid_split)
          wlocart_valid.fit(X_test, y_test, test_size = valid_test_size, min_samples_leaf = valid_min_sample)
          mean_valid_pred_set[it, 5], max_valid_pred_set[it, 5] = wlocart_valid.compute_dif()
      
          # smis
          smis_vector[it, 5] = smis(wlocart_pred, y_test, alpha = sig)
          
          # wsc
          wsc_vector[it, 5] = wsc_coverage(X_test, y_test, wlocart_pred)
          
          pcor_vector[it, 5] = pearson_correlation(wlocart_pred, y_test)
          HSIC_vector[it, 5] = HSIC_correlation(wlocart_pred, y_test)
      
          # mean interval length
          mean_int_length_vector[it, 5] = np.mean(compute_interval_length(wlocart_pred))
      
          # marginal coverage
          marg_cover = np.logical_and(y_test >= wlocart_pred[:, 0], 
              y_test <= wlocart_pred[:, 1]) + 0
          mean_coverage_vector[it, 5] = np.mean(
              marg_cover
          )
      
          # interval length | coveraqe
          cover_idx = np.where(marg_cover == 1)
          wlocart_interval_len_cover = np.mean(compute_interval_length(wlocart_pred[cover_idx]))
      
          wloc_cutoffs = wlocart_obj.cutoffs
      
      
          # fitting default regression split
          start_split = time.time()
          model = base_model(**kwargs)
          nc = NcFactory.create_nc(model)
          icp = IcpRegressor(nc)
          icp.fit(data["X_train"], data["y_train"])
          icp.calibrate(data["X_test"], data["y_test"])
      
          end_split = time.time() - start_split
          times[it, 6] = end_split

          # predictions
          icp_pred = icp.predict(X_test, significance = sig)
          icp_pred_cond = icp.predict(X_test, significance = sig)
          cond_icp_real = real_coverage(icp_pred_cond, y_mat)
          
          # average, median and max distance
          dif_icp = np.abs(cond_icp_real - (1 - sig))
          mean_diff_vector[it, 6], median_diff_vector[it, 6], max_diff_vector[it, 6] = (np.mean(dif_icp), 
          np.median(dif_icp), np.max(dif_icp))
      
          # valid pred sets
          icp_valid = Valid_pred_sets(conf = icp, alpha = sig, isnc = True, coverage_evaluator = "CART", 
          prune = valid_prune, split_train = valid_split)
          icp_valid.fit(X_test, y_test, test_size = valid_test_size, min_samples_leaf = valid_min_sample)
          mean_valid_pred_set[it, 6], max_valid_pred_set[it, 6] = icp_valid.compute_dif()
      
          # icp smis
          smis_vector[it, 6] = smis(icp_pred, y_test, alpha = sig)
          
          # wsc
          wsc_vector[it, 6] = wsc_coverage(X_test, y_test, icp_pred)
          
          pcor_vector[it, 6] = pearson_correlation(icp_pred, y_test)
          HSIC_vector[it, 6] = HSIC_correlation(icp_pred, y_test)
      
          # ICP interval length
          mean_int_length_vector[it, 6] = np.mean(compute_interval_length(icp_pred))
      
          # marginal coverage
          marg_cover = np.logical_and(y_test >= icp_pred[:, 0], 
              y_test <= icp_pred[:, 1]) + 0
          mean_coverage_vector[it, 6] = np.mean(
              marg_cover
          )
          
      
          # fitting wighted regression split
          start_weighted_split = time.time()
          wicp = LocalRegressionSplit(base_model, alpha = sig, **kwargs)
          wicp.fit(data["X_train"], data["y_train"], mad_model_cte = mad_model_cte)
          wicp.calibrate(data["X_test"], data["y_test"])
      
          end_weighted_split = time.time() - start_weighted_split
          times[it, 7] = end_weighted_split
      
          # predictions
          wicp_pred = wicp.predict(X_test)
          wicp_pred_cond = wicp.predict(X_test)
          cond_wicp_real = real_coverage(wicp_pred_cond, y_mat)
          
          
          wicp_dif = np.abs(cond_wicp_real - (1 - sig))
          mean_diff_vector[it, 7], median_diff_vector[it, 7], max_diff_vector[it, 7] = (np.mean(wicp_dif), 
          np.median(wicp_dif), np.max(wicp_dif))
      
          # valid pred sets
          wicp_valid = Valid_pred_sets(conf = wicp, alpha = sig, coverage_evaluator = "CART", prune = valid_prune, 
          split_train = valid_split)
          wicp_valid.fit(X_test, y_test, test_size = valid_test_size, min_samples_leaf = valid_min_sample)
          mean_valid_pred_set[it, 7], max_valid_pred_set[it, 7] = wicp_valid.compute_dif()
      
          # smis
          smis_vector[it, 7] = smis(wicp_pred, y_test, alpha = sig)
          
          # wsc
          wsc_vector[it, 7] = wsc_coverage(X_test, y_test, wicp_pred)
          
          pcor_vector[it, 7] = pearson_correlation(wicp_pred, y_test)
          HSIC_vector[it, 7] = HSIC_correlation(wicp_pred, y_test)
      
          # ICP interval length
          mean_int_length_vector[it, 7] = np.mean(compute_interval_length(wicp_pred))
      
          # marginal coverage
          marg_cover = np.logical_and(y_test >= wicp_pred[:, 0], 
              y_test <= wicp_pred[:, 1]) + 0
          mean_coverage_vector[it, 7] = np.mean(
              marg_cover
          )

          start_weighted_split = time.time()
          micp = MondrianRegressionSplit(base_model, alpha = sig, k = nbins, **kwargs)
          micp.fit(data["X_train"], data["y_train"])
          micp.calibrate(data["X_test"], data["y_test"])
      
          end_weighted_split = time.time() - start_weighted_split
          times[it, 8] = end_weighted_split
      
          # predictions
          micp_pred = micp.predict(X_test)
          micp_pred_cond = micp.predict(X_test)
          cond_micp_real = real_coverage(micp_pred_cond, y_mat)
          
          
          micp_dif = np.abs(cond_micp_real - (1 - sig))
          mean_diff_vector[it, 8], median_diff_vector[it, 8], max_diff_vector[it, 8] = (np.mean(micp_dif), 
          np.median(micp_dif), np.max(micp_dif))
      
          # valid pred sets
          micp_valid = Valid_pred_sets(conf = micp, alpha = sig, coverage_evaluator = "CART", prune = valid_prune, 
          split_train = valid_split)
          micp_valid.fit(X_test, y_test, test_size = valid_test_size, min_samples_leaf = valid_min_sample)
          mean_valid_pred_set[it, 8], max_valid_pred_set[it, 8] = micp_valid.compute_dif()
      
          # smis
          smis_vector[it, 8] = smis(micp_pred, y_test, alpha = sig)
          
          # wsc
          wsc_vector[it, 8] = wsc_coverage(X_test, y_test, micp_pred)
          
          pcor_vector[it, 8] = pearson_correlation(micp_pred, y_test)
          HSIC_vector[it, 8] = HSIC_correlation(micp_pred, y_test)
      
          # ICP interval length
          mean_int_length_vector[it, 8] = np.mean(compute_interval_length(micp_pred))
      
          # marginal coverage
          marg_cover = np.logical_and(y_test >= micp_pred[:, 0], 
              y_test <= micp_pred[:, 1]) + 0
          mean_coverage_vector[it, 8] = np.mean(
              marg_cover
          )
          
          if (it + 1) % 25 == 0 or (it + 1 == 1) or save_all:
            print("Saving data checkpoint on iteration {}".format(it + 1))
            # saving checkpoint of metrics
            saving_metrics(original_path, folder_path, var_path, kind, n_var,
            mean_int_length_vector, mean_diff_vector, median_diff_vector,
            max_diff_vector, mean_coverage_vector, smis_vector, wsc_vector, mean_valid_pred_set,
            max_valid_pred_set, pcor_vector, HSIC_vector, times)
    
        # saving all metrics again
        saving_metrics(original_path, folder_path, var_path, kind, n_var,
        mean_int_length_vector, mean_diff_vector, median_diff_vector,
        max_diff_vector, mean_coverage_vector, smis_vector, wsc_vector, mean_valid_pred_set,
        max_valid_pred_set, pcor_vector, HSIC_vector, times)
      
      else:
        continue
      
    print("Experiments finished for {} setting".format(kind))
    end_kind = time.time() - start_kind
    print("Time Elapsed to compute all metrics in the {} setting: {}".format(kind, end_kind))
    return end_kind

# saving metrics function
def saving_metrics(
  original_path, 
  folder_path, 
  var_path,
  kind,
  n_var,
  mean_int_length_vector,
  mean_diff_vector,
  median_diff_vector,
  max_diff_vector,
  mean_coverage_vector,
  smis_vector,
  wsc_vector,
  mean_valid_pred_set,
  max_valid_pred_set,
  pcor_vector, 
  HSIC_vector,
  times):
    # checking if path exsist
    if not(path.exists(original_path + folder_path + var_path)):
      # creating directory
      os.mkdir(original_path + folder_path + var_path)
 
    # changing working directory to the current folder
    os.chdir(original_path + folder_path + var_path)
      
    # saving all matrices into npy files
    # interval length
    np.save("mean_interval_length_p_{}_{}_data.npy".format(
      n_var, kind), mean_int_length_vector)
    
    # conditional difference
    np.save("mean_diff_p_{}_{}_data.npy".format(
      n_var, kind), mean_diff_vector)
    np.save("median_diff_p_{}_{}_data.npy".format(
      n_var, kind), median_diff_vector)
    np.save("max_diff_p_{}_{}_data.npy".format(
      n_var, kind), max_diff_vector)
      
    # mean coverage
    np.save("mean_coverage_p_{}_{}_data.npy".format(
      n_var, kind), mean_coverage_vector)
      
    # estimated metrics
    np.save("smis_p_{}_{}_data.npy".format(
      n_var, kind), smis_vector)
    np.save("wsc_p_{}_{}_data.npy".format(
      n_var, kind), wsc_vector)
    np.save("mean_valid_pred_set_p_{}_{}_data.npy".format(
      n_var, kind), mean_valid_pred_set)
    np.save("max_valid_pred_set_p_{}_{}_data.npy".format(
      n_var, kind), max_valid_pred_set)
    np.save("pcor_p_{}_{}_data.npy".format(
      n_var, kind), pcor_vector)
    np.save("HSIC_p_{}_{}_data.npy".format(
      n_var, kind), HSIC_vector)
      
    # running times
    np.save("run_times_p_{}_{}_data.npy".format(
      n_var, kind), times)
    
    # returning to original path
    os.chdir(original_path)
    
      
    


# method that make all the computations for all kinds of data

def compute_all_conformal_metrics(
  kinds_list = ["homoscedastic", "heteroscedastic", "asymmetric", "asymmetric_V2", "t_residuals", "non_cor_heteroscedastic"],
  base_model = RandomForestRegressor,
  completing = False,
  iter_completing = 50,
  save_all = True,
  n_it = 100,
  p = np.array([1,3,5]),
  d = 20,
  **kwargs):
    print("Starting all experiments")
    start_exp = time.time()
    times_list = list()
    if type(kinds_list) == list:
      for kinds in kinds_list:
        times_list.append(compute_metrics_sim(kind = kinds, n_it = n_it, completing = completing,
        iter_completing = iter_completing, save_all = save_all, p = p, d = d, **kwargs))
      end_exp = time.time() - start_exp
      print("Time elapsed to conduct all experiments: {}".format(end_exp))
      np.save("results/pickle_files/locart_all_metrics_experiments/sim_running_times.npy", np.array(times_list.append(end_exp)))
    else:
      compute_metrics_sim(kind = kinds_list, completing = completing,
        iter_completing = iter_completing, n_it = n_it, p = p, d = d, **kwargs)
      end_exp = time.time() - start_exp
      print("Time elapsed to conduct {} experiments: {}".format(kinds_list, end_exp))
      np.save("results/pickle_files/locart_all_metrics_experiments/sim_{}_running_times.npy".format(kinds_list), 
      np.array(times_list.append(end_exp)))
    return None

if __name__ == '__main__':
  print("We will now compute all conformal statistics for several simulated examples")
  model = input("Which model would like to use as base model? ")
  separated = input("Would you like to run each setting in separated terminals? ")
  if separated == "yes":
    kind = input("What kind of data would you like to simulate? ")
  if model == "Random Forest":
    random_state = 650
    if separated == "yes":
      compute_all_conformal_metrics(kinds_list = kind, random_state = random_state)
    else:
      compute_all_conformal_metrics(random_state = random_state)
  elif model == "KNN":
    if separated == "yes":
      compute_all_conformal_metrics(kinds_list = kind, base_model = KNeighborsRegressor, n_neighbors = 30)
    else:
      compute_all_conformal_metrics(base_model = KNeighborsRegressor, n_neighbors = 30)
        
  
        
        
        
        
        
        
        
    
    
      

