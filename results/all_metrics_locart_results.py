# formatting raw data into new csv file
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os import path
import pandas as pd

# 9 columns in raw data
# first column is normal locart
# second column is random forest locart
# third is difficulty locart
# forth is difficulty random forest locart
# fifth is LCP-RF
# sixth is weighted locart
# seventh is ICP
# eighth is WICP
# and nineth is mondrian split

original_path = os.getcwd()
# plotting object (if needed)
plt.style.use('ggplot')
sns.set_palette("Set1")

# creating csv file with all needed data
def create_data_list(kind, 
p = np.array([1, 3, 5]),
d = 20,
n_it = 100,
type_mod = "regression",
exp_path = "/results/pickle_files/locart_all_metrics_experiments/",
other_asym = False):
  
  # folder path
  folder_path = exp_path + kind + "_data"
  print(folder_path)
  if other_asym:
    folder_path = folder_path + "_eta_1.5"
  
  # strings names to be used
  max_diff, mean_diff, median_diff = "max_diff", "mean_diff", "median_diff"
  mean_int, mean_coverage = "mean_interval_length", "mean_coverage"
  mean_valid, max_valid = "mean_valid_pred_set", "max_valid_pred_set"
  hsic, pcor, smis, wsc = "HSIC", "pcor", "smis", "wsc"
  times = "run_times"
  
  # name of each method
  methods = ["locart", "RF-locart", "Dlocart", "RF-Dlocart", "LCP-RF", "Wlocart", "icp", "wicp", "mondrian"]
  
  string_names = [mean_diff, median_diff, max_diff, smis, wsc, hsic, pcor, mean_valid, max_valid, mean_int, mean_coverage, times]
  
  # checking if path exists and then importing all data matrices
  if path.exists(original_path + folder_path):
    print("Creating data list")
    # list of data frames
    stat_list = list()
    for i in range(p.shape[0]):
      # importing the data
      current_folder = original_path + folder_path + "/{}_score_regression_p_{}_10000_samples_measures".format(
        kind, p[i])
      
      # looping through all string names
      data_list = []
      for string in string_names:
        current_data = np.load(current_folder + "/" + string + "_p_{}_{}_data.npy".format(p[i], kind))
        # removing rows with only zeroes
        current_data = current_data[~np.all(current_data == 0, axis = 1)]
        data_list.append(current_data)
      
      # obtaining mean vectors in each matrix
      means_list = [np.mean(data, axis = 0) for data in data_list]
      # standard deviation
      sd_list = [np.std(data, axis = 0) for data in data_list]
      
      # transforming means_list and sd_list into a matrix
      means_array = np.column_stack(means_list)
      
      # transforming sd into a flat array
      sd_array = np.concatenate(sd_list)
      
      # transforming into a pandas dataframe and adding a new variable identifying the n and the methods
      # and melting it to add varible column
      new_data = (pd.DataFrame(means_array,
      columns = string_names).
      assign(p_var = p[i],
      methods = methods).
      melt(id_vars = ["n", "methods"],
      value_vars = string_names,
      var_name = "stats").
      assign(sd = sd_array/np.sqrt(n_it)))
      
      # adding to a list of data frames
      stat_list.append(new_data)
      
    # concatenating the data frames list into a single one data and saving it to csv
    data_final = pd.concat(stat_list)
    # saving to csv and returning
    data_final.to_csv(folder_path + "/stats.csv")
    return(data_final)
  
  
