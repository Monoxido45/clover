# plotting all the results from each kind of simulated data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os import path
import pandas as pd

# first column is locluster
# second is locart
# third is ICP
# fourth is WICP
# and fifth is euclidean binning

original_path = os.getcwd()
# for now, we will plot for homoscedastic, heteroscedastic and asymmetric data
# function to plot each kind of data
plt.style.use('ggplot')
sns.set_palette("Set1")

def create_data_list(kind, 
ntrain = np.array([500, 1000, 5000, 10000]),
d = 20,
n_it = 200,
type_mod = "regression",
exp_path = "/results/pickle_files/locart_experiments_results/",
other_asym = False):
  
  # folder path
  folder_path = exp_path + kind + "_data"
  print(folder_path)
  if other_asym:
    folder_path = folder_path + "_eta_1.5"
  
  # strings names to be used
  median_str, median_diff, median_coverage = "median_interval_length", "median_diff", "median_coverage" 
  mean_str, mean_diff, mean_coverage = "mean_interval_length", "mean_diff", "mean_coverage"
  
  # name of each method
  methods = ["locluster", "locart", "icp", "wicp", "euclidean", "mondrian"]
  
  string_names = [median_str, median_diff, median_coverage, 
  mean_str, mean_diff, mean_coverage]
  
  # checking if path exists and then importing all data matrices
  if path.exists(original_path + folder_path):
    print("Creating data list")
    stat_list = list()
    for i in range(ntrain.shape[0]):
      # importing the data
      current_folder = original_path + folder_path + "/{}_score_{}_dim_{}_{}_samples_measures".format(
        kind, type_mod, d, ntrain[i])
      
      # list containing all data for n = ntrain[i]
      data_list = [np.column_stack((np.load(current_folder + "/" + string + "_n_{}_{}_data.npy".format(ntrain[i], kind)),
      np.load(current_folder + "/mondrian/" + string + "_n_{}_{}_data.npy".format(ntrain[i], kind)))) for string in string_names]
      
      # obtaining mean vectors in each matrix
      means_list = [np.mean(data, axis = 0) for data in data_list]
      # standard deviation
      sd_list = [np.std(data, axis = 0) for data in data_list]
      
      # transforming means_list into a matrix
      means_array = np.column_stack(means_list)
      
      # transforming sd into a flat array
      sd_array = np.concatenate(sd_list)
      
      # transforming into a pandas dataframe and adding a new variable identifying the n and the methods
      # and melting it to add varible column
      new_data = (pd.DataFrame(means_array,
      columns = string_names).
      assign(n = ntrain[i],
      methods = methods).
      melt(id_vars = ["n", "methods"],
      value_vars = string_names,
      var_name = "stats").
      assign(sd = sd_array/np.sqrt(n_it)))
      
      # adding to a list of data frames
      stat_list.append(new_data)
    return(stat_list)
      
      
def import_all_results(kind, 
ntrain = np.array([500, 1000, 5000, 10000]),
d = 20,
n_it = 200,
type_mod = "regression",
images_dir = "results/figures", 
exp_path = "/results/pickle_files/locart_experiments_results/",
other_asym = False):
  # first creating the data list to be plotted
  data_list = create_data_list(kind, ntrain, d, n_it, type_mod, exp_path, other_asym)
  
  # concatenating data list
  data_final = pd.concat(data_list)
  if other_asym:
    figname = "{}_data_eta_1.5_{}_experiments".format(kind, type_mod)
  else:
    figname = "{}_data_{}_experiments".format(kind, type_mod)
  # with the final data in hands, we can plot the line plots as desired
  # faceting all in a seaborn plot
  g = sns.FacetGrid(data_final, col = "stats", col_wrap = 3, hue = "methods",
  despine = False, margin_titles = True, legend_out = False,
  sharey = False,
  height = 5)
  g.map(plt.errorbar, "n", "value", "sd", marker = "o")
  g.figure.subplots_adjust(wspace=0, hspace=0)
  g.add_legend()
  g.set_ylabels("Values")
  g.set_xlabels("Number of training/calibration samples")
  plt.tight_layout()
  plt.savefig(f"{images_dir}/{figname}.pdf")
  plt.show()
  
  
  
  
