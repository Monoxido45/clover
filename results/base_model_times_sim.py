# saving the time each base model runs for each setting
import numpy as np
import pandas as pd
import os
from os import path

# base models
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# simulation and performance measures
import time
from lcv.simulation import simulation

original_path = os.getcwd()

def run_base_model_sim(
  kind = "homoscedastic",
  n_it = 100,
  n = 10000,
  p = np.array([1, 3, 5]),
  d = 20,
  hetero_value = 0.25,
  asym_value = 0.6,
  t_degree = 4,
  base_model = RandomForestRegressor, 
  coef = 2,
  random_seed = 1250,
  **kwargs):
     # starting experiment
    print("Starting base model running time for {} data with p = {}".format(kind, p))
    if "_V2" in kind:
      asym_value = 1.5
      kind = "asymmetric"
      folder_path = "/results/pickle_files/locart_all_metrics_experiments/{}_data_eta_{}".format(
      kind, asym_value)
    else:
      folder_path = "/results/pickle_files/locart_all_metrics_experiments/{}_data".format(
        kind)
    
    np.random.seed(random_seed)
    random_seeds = np.random.randint(0, 10**(8), n_it)
        
    for n_var in p:
      var_path = "/{}_score_regression_p_{}_{}_base_model_time".format(kind, n_var, n)
      noise = (n_var == 1)
      if not(path.exists(original_path + folder_path + var_path)):
        running_time = np.zeros(n_it)
        print("running the experiments for {} significant variables and {} training instances in the {} setting".format(n_var, 
        n, kind))
        for it in range(0,n_it):
          if (it + 1) % 25 == 0:
            print("Running iteration {}".format(it + 1))
          # simulating data
          sim_obj = simulation(dim = d, coef = coef, hetero_value = hetero_value, 
          noise = noise, signif_vars = n_var, asym_value = asym_value, t_degree = t_degree)
          r_kind = getattr(sim_obj, kind + "_r")
          sim_kind = getattr(sim_obj, kind)
          sim_kind(n, random_seed = random_seeds[it])
          
          start_loc = time.time()
          
          model = base_model(**kwargs)
          model.fit(sim_obj.X, sim_obj.y)
          
          end_loc = time.time() - start_loc
          running_time[it] = end_loc
        
        # saving running times
        if not(path.exists(original_path + folder_path + var_path)):
          # creating directory
          os.mkdir(original_path + folder_path + var_path)
        
         # changing working directory to the current folder
        os.chdir(original_path + folder_path + var_path)
      
        # saving all matrices into npy files
        # interval length
        np.save("model_running_time_p_{}_{}_data.npy".format(
                  n_var, kind), running_time)
        # returning to original path
        os.chdir(original_path)
    
    return None


if __name__ == '__main__':
  print("We will now compute all base model running time for several simulated examples")
  model = input("Which model would like to use as base model? ")
  kind = input("What kind of data would you like to simulate? ")
  if model == "Random Forest":
    random_state = 650
    run_base_model_sim(kind = kind, random_state = random_state)
          
        

