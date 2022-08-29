# script to compute LCV hypothesis test power for random forest coverage evaluator with B = 500
# importing functions from lcv package
from lcv.valid_pred_sets import Valid_pred_sets
from lcv.valid_pred_sets import LinearQuantileRegression
from lcv.valid_pred_sets import GradientBoostingQuantileRegression 
import numpy as np
import os
from os import path
import time
# importing simulation
path_original = os.getcwd()
os.chdir(path_original + "/results")
from simulation import simulation

# returning to original path
os.chdir(path_original)

# function to compute power of the test
# using kind = "bimodal" as default

def compute_power(kind = "bimodal", npower = 100, B = 350, ntrain = np.arange(1000, 25000, 5000), coverage_evaluator = "RF",
ncalib = 750, par = True, type_reg = "gb_quantile", random_seed = 1250, sig = 0.05, rej = 0.05, d = 50):
  np.random.seed(random_seed)
  print("running power experiments for {} regression and {} data".format(type_reg, kind))
  for i in range(ntrain.shape[0]):
    power = np.zeros(1)
    if not(path.exists("results/pickle_files/power_tests_{}_n_{}_d_{}_{}_{}_data.npy".format(type_reg, ntrain[i], d, coverage_evaluator, kind))):
      print("running {} experiment with {} samples".format(i + 1, ntrain[i]))
      seeds_calib, seeds_train = np.random.randint(1e8, size = npower), np.random.randint(1e8, size = npower)
      seeds_gbqr, seeds_split = np.random.randint(1e8, size = npower), np.random.randint(1e8, size = npower)
      p_values = np.zeros(npower)
      for j in range(npower):
        # simulating calibration and training samples
        sim_obj = simulation(dim = d)
        sim_kind = getattr(sim_obj, kind)
        sim_kind(ncalib, random_seed = seeds_calib[j])
        
        # using gradient boosting quantile regression
        if type_reg == "gb_quantile":
          gbqr = GradientBoostingQuantileRegression(n_estimators = 200, random_state = seeds_gbqr[j])
          gbqr.fit(sim_obj.X, sim_obj.y)
          hyp_gbqr_quant = Valid_pred_sets(gbqr, sig, coverage_evaluator = coverage_evaluator)
          # now generating the training samples
          sim_kind(ntrain[i], random_seed = seeds_train[j])
          hyp_gbqr_quant.fit(sim_obj.X, sim_obj.y, random_seed = seeds_split[j])
          # checking if object prints
          print(hyp_gbqr_quant)
          p_values[j] = hyp_gbqr_quant.monte_carlo_test(B = B, random_seed = random_seed, par = par)["p-value"]
          print(p_values[j])
          
        # using linear quantile regression
        elif type_reg == "l_quantile":
          lqr = LinearQuantileRegression(alpha = 1, solver = "highs")
          lqr.fit(sim_obj.X, sim_obj.y)
          hyp_lqr_quant = Valid_pred_sets(lqr, sig, coverage_evaluator = coverage_evaluator)
          # generating calibration
          sim_kind(ntrain[i], random_seed = seeds_train[j])
          hyp_lqr_quant.fit(sim_obj.X, sim_obj.y, random_seed = seeds_split[j])
          p_values[j] = hyp_lqr_quant.monte_carlo_test(B = B, random_seed = random_seed, par = par)["p-value"]
          
      # computing power and saving for each different number of training samples
      print("pickling up everything")
      power[0] = np.mean(p_values <= rej)
      np.save("results/pickle_files/power_tests_{}_n_{}_d_{}_{}_{}_data.npy".format(type_reg, ntrain[i], d, coverage_evaluator, kind), power)
      print("finished {} experiment".format(i + 1))
    else:
      continue
  return None

# running the script
if __name__ == '__main__':
  start = time.time()
  compute_power()
  end = time.time() - start
  print("Time Elapsed: ", end)


