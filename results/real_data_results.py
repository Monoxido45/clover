# formatting raw data into new csv file
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os import path
import pandas as pd
from pandas.api.types import CategoricalDtype

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
plt.style.use('seaborn-white')
sns.set_palette("Set1")

# creating csv file with all needed data
def create_data_list(data_name, 
n_it = 100,
exp_path = "/results/pickle_files/real_data_experiments/"):
  
  # folder path
  folder_path = exp_path + data_name + "_data"
  print(folder_path)
  
  # strings names to be used
  mean_int, mean_int_cover, mean_coverage = "mean_interval_length", "mean_interval_length_cover", "mean_coverage"
  mean_valid, max_valid = "mean_valid_pred_set", "max_valid_pred_set"
  hsic, pcor, smis = "HSIC", "pcor", "smis"
  
  # name of each method
  methods = ["locart", "RF-locart", "Dlocart", "RF-Dlocart", "LCP-RF", "Wlocart", "icp", "wicp", "mondrian"]
  
  string_names = [smis, hsic, pcor, mean_valid, max_valid, mean_int, mean_int_cover, mean_coverage]
  
  # checking if path exists and then importing all data matrices
  if path.exists(original_path + folder_path):
    print("Creating data frames")
    
    # importing the data
    current_folder = original_path + folder_path + "/{}_data_score_regression_measures".format(
      data_name)
    
    # looping through all string names
    data_list = []
    corr_data_list = []
    for string in string_names:
      current_data = np.load(current_folder + "/" + string + "_{}_data.npy".format(data_name))
      # removing rows with only zeroes
      current_data = current_data[~np.all(current_data == 0, axis = 1)]
      if string == "smis":
        current_data = - current_data
      # data list for extracting means  
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
    data_final = (pd.DataFrame(means_array,
    columns = string_names).
    assign(methods = methods).
    melt(id_vars = ["methods"],
    value_vars = string_names,
    var_name = "stats").
    assign(sd = sd_array/np.sqrt(n_it)))

    # saving to csv and returning
    data_final.to_csv(original_path + folder_path + "/{}_stats.csv".format(data_name))
    
    return(data_final)
  
  
  
def create_data_times(data_name,
exp_path = "/results/pickle_files/real_data_experiments/"):
    # folder path
  folder_path = exp_path + data_name + "_data"
  print(folder_path)
  
  string = "run_times"
  
  # name of each method
  methods = ["locart", "RF-locart", "Dlocart", "RF-Dlocart", "LCP-RF", "Wlocart", "icp", "wicp", "mondrian"]
  
  # checking if path exists and then importing all data matrices
  if path.exists(original_path + folder_path):
    print("Creating data list")
    # list of data frames
    data_list = list()
      # importing the data
    current_folder = original_path + folder_path + "/{}_data_score_regression_measures".format(data_name)
    
    # only one string name
    current_data = np.load(current_folder + "/" + string + "_{}_data.npy".format(data_name))
    
    # removing rows with only zeroes
    current_data = current_data[~np.all(current_data == 0, axis = 1)]
    
    data_final = (pd.DataFrame(current_data,
    columns = methods).
    melt(
    value_vars = methods,
    var_name = "methods"))

    # saving to csv and returning
    data_final.to_csv(original_path + folder_path + "/{}_running_times.csv".format(data_name))
    return(data_final)
  
  
# saving csv files for some of the real data
data_names = ["winewhite", "winered",  "concrete", "airfoil", "electric", "superconductivity", "cycle",
"protein", "news"]

for data in data_names:
  created_data = create_data_list(data)
  created_data_times = create_data_times(data)
  
  
# function to generate graphs
def plot_results(data_names_list,
images_dir = "results/metric_figures",
vars_to_plot = np.array(["smis"]),
exp_path = "/results/pickle_files/real_data_experiments/"):
  figname = "performance_real/smis_performance"
  data_list = []
  for name in data_names_list:
    folder_path = exp_path + name + "_data"
    # first creating the data list to be plotted
    current_data = (pd.read_csv(original_path + folder_path + "/{}_stats.csv".format(name)).
    assign(sd = lambda df: df['sd']*2).
    assign(data = name))
    data_list.append(current_data)
  
  data_main = pd.concat(data_list)
  
  # ordering data by custom order
  method_custom_order = CategoricalDtype(
    ["locart", "Dlocart", "RF-locart", "RF-Dlocart", "Wlocart", "LCP-RF", "icp", "wicp", "mondrian"], 
    ordered=True)
  
  # sorting according to the custom order
  data_main['methods'] = data_main['methods'].astype(method_custom_order)
  
  # with the final data in hands, we can plot the line plots as desired
  # faceting all in a seaborn plot
  g = sns.FacetGrid(data_main.
  query("stats in @vars_to_plot"), col = "data", col_wrap = 3, hue = "methods",
  despine = False, margin_titles = True, legend_out = False,
  sharey = False,
  height = 5)
  g.map(plt.errorbar, "methods", "value", "sd", marker = "o")
  g.figure.subplots_adjust(wspace=0, hspace=0)
  g.add_legend()
  g.set_ylabels("Metric values")
  g.set_xlabels("Methods")
  g.set_xticklabels(rotation = 45)
  plt.tight_layout()
  plt.savefig(f"{images_dir}/{figname}.pdf")
  
  # returning data final to plot more general graphs
  return(data_main)

plot_results(data_names)


