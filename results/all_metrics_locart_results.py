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
def create_data_list(kind, 
p = np.array([1, 3, 5]),
n_it = 100,
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
  
  # name of each method
  methods = ["locart", "RF-locart", "Dlocart", "RF-Dlocart", "LCP-RF", "Wlocart", "icp", "wicp", "mondrian"]
  
  string_names = [mean_diff, median_diff, max_diff, smis, wsc, hsic, pcor, mean_valid, max_valid, mean_int, mean_coverage]
  
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
        if string == "smis":
          current_data = - current_data
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
      melt(id_vars = ["p_var", "methods"],
      value_vars = string_names,
      var_name = "stats").
      assign(sd = sd_array/np.sqrt(n_it)))
      
      # adding to a list of data frames
      stat_list.append(new_data)
      
    # concatenating the data frames list into a single one data and saving it to csv
    data_final = pd.concat(stat_list)
    # saving to csv and returning
    data_final.to_csv(original_path + folder_path + "/{}_stats.csv".format(kind))
    return(data_final)


def create_data_times(kind, 
p = np.array([1, 3, 5]),
exp_path = "/results/pickle_files/locart_all_metrics_experiments/",
other_asym = False):
    # folder path
  folder_path = exp_path + kind + "_data"
  print(folder_path)
  if other_asym:
    folder_path = folder_path + "_eta_1.5"
  
  string = "run_times"
  
  # name of each method
  methods = ["locart", "RF-locart", "Dlocart", "RF-Dlocart", "LCP-RF", "Wlocart", "icp", "wicp", "mondrian"]
  
  # checking if path exists and then importing all data matrices
  if path.exists(original_path + folder_path):
    print("Creating data list")
    # list of data frames
    data_list = list()
    for i in range(p.shape[0]):
      # importing the data
      current_folder = original_path + folder_path + "/{}_score_regression_p_{}_10000_samples_measures".format(
        kind, p[i])
      
      # only one string name
      current_data = np.load(current_folder + "/" + string + "_p_{}_{}_data.npy".format(p[i], kind))
      # removing rows with only zeroes
      current_data = current_data[~np.all(current_data == 0, axis = 1)]
      new_data = (pd.DataFrame(current_data,
      columns = methods).
      assign(p_var = p[i]).
      melt(id_vars = ["p_var"],
      value_vars = methods,
      var_name = "methods"))
      
      # adding to a list of data frames
      data_list.append(new_data)
      
    # concatenating the data frames list into a single one data and saving it to csv
    data_final = pd.concat(data_list)
    # saving to csv and returning
    data_final.to_csv(original_path + folder_path + "/{}_running_times.csv".format(kind))
    return(data_final)
  
# creating data_list to several kind of data
def create_all_data(kind_list = ["homoscedastic", "heteroscedastic", "asymmetric", 
"asymmetric_V2", "t_residuals", "non_cor_heteroscedastic"],
p = np.array([1,3,5]),
n_it = 100,
times = False):
  data_list = []
  for kind in kind_list:
    if kind == "asymmetric_V2":
      other_asym = True
      kind = "asymmetric"
    else:
      other_asym = False
      
    if times:
      # assigning type of data
      data = (create_data_times(kind, p = p, other_asym = other_asym).
      assign(data_type = kind))
      
    else:
      # assigning the type of data
      data = (create_data_list(kind, p = p, n_it = n_it, other_asym = other_asym).
      assign(data_type = kind))
      
    data_list.append(data)
  return data_list


# saving several data at the same time and generating a list of data
data_list = create_all_data(p = np.array([1, 3, 5]))
data_time = create_all_data(p = np.array([1, 3, 5]), times = True)

def plot_results_by_methods(kind, 
p = np.array([1,3,5]),
images_dir = "results/metric_figures",
vars_to_plot = np.array(["mean_diff", "smis", "wsc", "mean_valid_pred_set"]),
vars_corr = np.array(["smis", "mean_valid_pred_set", "wsc", "pcor", "HSIC", "mean_diff", "max_diff"]),
other_vars = np.array(["mean_coverage", "mean_interval_length"]),
exp_path = "/results/pickle_files/locart_all_metrics_experiments/",
other_asym = False):
  if other_asym:
    figname = "sim_{}_data_eta_1.5_regression_experiments".format(kind)
    fig_times = "sim_{}_data_eta_1.5_running_times".format(kind)
    fig_corr = "sim_{}_data_eta_1.5_corr_mat".format(kind)
    figname_others = "sim_{}_data_eta_1.5_other_measures".format(kind)
    folder_path = exp_path + kind + "_data_eta_1.5"
  else:
    figname = "sim_{}_data_regression_experiments".format(kind)
    fig_times = "sim_{}_data_running_times".format(kind)
    fig_corr = "sim_{}_data_corr_mat".format(kind)
    figname_others = "sim_{}_data_other_measures".format(kind)
    folder_path = exp_path + kind + "_data"
    
   # first creating the data list to be plotted
  data = (pd.read_csv(original_path + folder_path + "/{}_stats.csv".format(kind)).
  assign(sd = lambda df: df['sd']*2).
  query("p_var in @p").
  query("stats in @vars_to_plot"))
  
  # ordering data by custom order
  method_custom_order = CategoricalDtype(
    ["locart", "Dlocart", "RF-locart", "RF-Dlocart", "Wlocart", "LCP-RF", "icp", "wicp", "mondrian"], 
    ordered=True)
  
  # sorting according to the custom order
  data['methods'] = data['methods'].astype(method_custom_order)
  data = data.sort_values('methods')
  
  # with the final data in hands, we can plot the line plots as desired
  # faceting all in a seaborn plot
  g = sns.FacetGrid(data, col = "stats", row = "p_var", hue = "methods",
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
  
  # importing running times data
  data_times = (pd.read_csv(original_path + folder_path + "/{}_running_times.csv".format(kind)).
  query("p_var in @p"))
  data_times['methods'] = data_times['methods'].astype(method_custom_order)
  data_times = data_times.sort_values('methods')
  
  # plotting running times as boxplot
  plt.figure(figsize = (16, 8))
  sns.boxplot(data = data_times, x = 'methods', y = 'value', hue = 'methods')
  plt.xlabel("Methdos")
  plt.ylabel("Running times (seconds)")
  plt.xticks(rotation = 45)
  plt.savefig(f"{images_dir}/{fig_times}.pdf")
  
  # plotting heatmap with correlations for all p
  # creating data to be plotted
  cor_matrix = (pd.read_csv(original_path + folder_path + "/{}_stats.csv".format(kind)).
  query("p_var in @p").
  query("stats in @vars_corr").
  loc[:, ['p_var', 'methods', 'stats', 'value']].
  pivot(index = ['p_var', 'methods'], columns = 'stats',
  values = 'value').
  corr(method = "spearman"))
  
  plt.figure(figsize = (16, 8))
  sns.heatmap(cor_matrix, 
        xticklabels = cor_matrix.columns,
        yticklabels = cor_matrix.columns,
        annot=True,
        cmap = "Blues")
  plt.savefig(f"{images_dir}/{fig_corr}.pdf")
  
  # verifying mean marginal coverage with 
  data = (pd.read_csv(original_path + folder_path + "/{}_stats.csv".format(kind)).
  assign(sd = lambda df: df['sd']*2).
  query("p_var in @p").
  query("stats in @other_vars"))
  
  # sorting according to the custom order
  data['methods'] = data['methods'].astype(method_custom_order)
  data = data.sort_values('methods')
  
  # with the final data in hands, we can plot the line plots as desired
  # faceting all in a seaborn plot
  g = sns.FacetGrid(data, col = "stats", row = "p_var", hue = "methods",
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
  plt.savefig(f"{images_dir}/{figname_others}.pdf")
  

# plotting all at the same time
if __name__ == '__main__':
  print("plotting all results for wsc, smis and real diff")
  # selecting all p's
  p = np.array([1,3, 5])
  kinds_list = ["homoscedastic", "heteroscedastic", "asymmetric", "asymmetric_V2", "t_residuals", "non_cor_heteroscedastic"]
  for kind in kinds_list:
    if kind == "asymmetric_V2":
      other_asym = True
      kind = "asymmetric"
      plot_results_by_methods(kind, p = p, other_asym = other_asym)
    else:
      plot_results_by_methods(kind, p = p)
    
