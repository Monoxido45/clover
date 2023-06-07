# formatting raw data into new csv file
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os import path
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.ticker as tkr

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
sns.set_palette("tab10")
plt.rcParams.update({'font.size': 12})

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
  methods = ["locart", "loforest", "A-locart", "A-loforest", "QRF-TC", "W-loforest", "reg-split", "W-reg-split", 
  "mondrian"]
  
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
      
      # data list just to melt everything and compute data frames
      
    
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
  methods = ["locart", "loforest", "A-locart", "A-loforest", "QRF-TC", "W-loforest", "reg-split", 
  "W-reg-split", "mondrian"]
  
  # checking if path exists and then importing all data matrices
  if path.exists(original_path + folder_path):
    print("Creating data list")
    # list of data frames
    data_list = list()
      # importing the data
    current_folder = original_path + folder_path + "/{}_data_score_regression_measures".format(data_name)
    
    # importing the correction
    correction_folder = original_path + folder_path + "/{}_data_score_regression_model_time".format(data_name)
    
    # only one string name
    current_data = np.load(current_folder + "/" + string + "_{}_data.npy".format(data_name))
    correction_data = np.load(correction_folder + "/" + "model_running_time_{}_data.npy".format(data_name))
    
    # removing rows with only zeroes
    current_data = current_data[~np.all(current_data == 0, axis = 1)]
    
    current_data[:,6] = np.abs(current_data[:,6] - correction_data)
    
    # obtaining proportions by row
    prop_data = current_data/np.sum(current_data, axis = 1)[:, None]
    
    # proportion data
    data_prop = (pd.DataFrame(prop_data,
    columns = methods).
    melt(
      value_vars = methods,
      var_name = "methods"
    ))
    
    # melted data
    data_final = (pd.DataFrame(current_data,
    columns = methods).
    melt(
    value_vars = methods,
    var_name = "methods"))
    

    # saving to csv and returning
    data_final.to_csv(original_path + folder_path + "/{}_running_times.csv".format(data_name))
    data_prop.to_csv(original_path + folder_path + "/{}_prop_times.csv".format(data_name))
    return(data_final)
  
  
# saving csv files for some of the real data
data_names = ["winewhite", "winered",  "concrete", "airfoil", "electric", "superconductivity", "cycle",
"protein", "news", "bike", "star", "meps19"]

for data in data_names:
  created_data = create_data_list(data)
  created_data_times = create_data_times(data)
  
  
# function to generate graphs
def plot_results(data_names_list,
images_dir = "results/metric_figures",
vars_to_plot = np.array(["smis"]),
exp_path = "/results/pickle_files/real_data_experiments/"):
  
  figname = "performance_real/smis_performance"
  marginal_cover = "performance_real/marginal_coverage"
  barplot_res = "performance_real/smis_barplot"
  run_time_prop = "performance_real/running_time_plot"
  
  our_methods = ["locart", "A-locart", "loforest", "A-loforest", "W-loforest"]
  data_list = []
  time_data_list = []
  for name in data_names_list:
    folder_path = exp_path + name + "_data"
    # first creating the data list to be plotted
    current_data = (pd.read_csv(original_path + folder_path + "/{}_stats.csv".format(name)).
    assign(sd = lambda df: df['sd']*2).
    assign(data = name))
    data_list.append(current_data)
    
    # time data
    time_data = (pd.read_csv(original_path + folder_path + "/{}_prop_times.csv".format(name)).
    assign(data = name))
    time_data_list.append(time_data)
  
  data_main = pd.concat(data_list)
  time_data = pd.concat(time_data_list)
  
  # ordering data by custom order
  method_custom_order = CategoricalDtype(
    ["locart", "A-locart", "reg-split", "W-reg-split", 
    "mondrian", "loforest", "A-loforest", "W-loforest", "QRF-TC"], 
    ordered=True)
    
  conf_methods = ["locart", "A-locart", "reg-split", "W-reg-split", "mondrian"]
  
  # sorting according to the custom order
  data_main['methods'] = data_main['methods'].astype(method_custom_order)
  data_main = data_main.assign(conformal = lambda df: df['methods'].map(
    lambda methods: True if methods in conf_methods else False))
  time_data['methods'] = time_data['methods'].astype(method_custom_order)
  
  
  # with the final data in hands, we can plot the line plots as desired
  # faceting all in a seaborn plot
  g = sns.FacetGrid(data_main.
  query("stats in @vars_to_plot"), col = "data", col_wrap = 3, hue = "methods",
  despine = False, margin_titles = True, legend_out = True,
  sharey = False,
  height = 5)
  g.map(plt.errorbar, "methods", "value", "sd", marker = "o")
  g.map(plt.axvline, x = 4.5, color = 'k', linestyle = "dashed")
  g.figure.subplots_adjust(wspace=0, hspace=0)
  g.add_legend(bbox_to_anchor = (1.1, 0.5), title = "Methods")
  legend = g.legend
  for text in legend.get_texts():
    if text.get_text() in our_methods:
      text.set_fontweight("bold")
      
  g.set_ylabels("SMIS values")
  g.set_xlabels("Methods")
  g.set_xticklabels(rotation = 45)
  g.set_titles(col_template="{col_name}")
  
  # transforming some labels into bold
  for ax in g.axes.flatten():
    if len(ax.get_xticklabels()) != 0:
      for idx in [0,1,5,6,7]:
        ax.get_xticklabels()[idx].set_fontweight("bold")
      
  plt.tight_layout()
  plt.savefig(f"{images_dir}/{figname}.pdf", bbox_inches="tight")
  
  # plotting methods with best performance in barplot
  # now with better smis
  data_count_smis_all = (data_main.
  query("stats == 'smis'").
  groupby(['data']).
  apply(lambda df: df.nsmallest(n = 1, columns = 'value', keep = "all")).
  value_counts("methods"))
  
  data_count_smis_conf = (data_main.
  query("stats == 'smis'").
  query("conformal == True").
  assign(methods = lambda df: df.methods.values.remove_categories(["loforest", 
  "A-loforest", "QRF-TC", "W-loforest"])).
  groupby(['data']).
  apply(lambda df: df.nsmallest(n = 1, columns = 'value', keep = "all")).
  value_counts("methods"))
  
  data_count_smis_no_conf = (data_main.
  query("stats == 'smis'").
  query("conformal == False").
   assign(methods = lambda df: df.methods.values.remove_categories(["locart", "A-locart", 
   "reg-split", "W-reg-split", "mondrian"])).
  groupby(['data']).
  apply(lambda df: df.nsmallest(n = 1, columns = 'value', keep = "all")).
  value_counts("methods"))
  
  # plotting count of data into two barplots
  fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (12, 6))
  ax1.bar(x = data_count_smis_conf.keys(), 
  height = data_count_smis_conf.values,
  color = "tab:blue", alpha = 0.5)
  ax1.set_title("Conformal methods")
  ax1.set_xlabel("Methods")
  ax1.set_ylabel("Frequency")
  ax1.tick_params(axis = "x", labelrotation=45)
  for idx in [0,4]:
    ax1.get_xticklabels()[idx].set_fontweight("bold")

  ax2.bar(x = data_count_smis_no_conf.keys(), 
  height = data_count_smis_no_conf.values,
  color = "tab:blue", alpha = 0.5)
  ax2.set_title("Non conformal methods")
  ax2.set_xlabel("Methods")
  ax2.set_ylabel("Frequency")
  ax2.tick_params(axis = "x", labelrotation=45)
  for idx in [0,2,3]:
    ax2.get_xticklabels()[idx].set_fontweight("bold")
  
  ax3.bar(x = data_count_smis_all.keys(), 
  height = data_count_smis_all.values,
  color = "tab:blue", alpha = 0.5)
  ax3.set_title("All methods")
  ax3.set_xlabel("Methods")
  ax3.set_ylabel("Frequency")
  ax3.tick_params(axis = "x", labelrotation=45)
  for idx in [0,2,3,4,8]:
    ax3.get_xticklabels()[idx].set_fontweight("bold")
  
  plt.tight_layout()
  plt.savefig(f"{images_dir}/{barplot_res}.pdf")
  
  # plotting time proportions boxplot
  plt.figure(figsize = (12, 8))
  sns.barplot(data = time_data, x = "data", y = "value", hue = "methods", 
  dodge = True)
  legend = plt.legend(bbox_to_anchor = (1.025, 0.5), title = "Methods")
  for text in legend.get_texts():
    if text.get_text() in our_methods:
      text.set_fontweight("bold")
  plt.xlabel("Data")
  plt.ylabel("Running time proportions (log scale)")
  plt.yscale('log')
  plt.xticks(rotation = 45)
  plt.savefig(f"{images_dir}/{run_time_prop}.pdf", bbox_inches="tight")
  
  # plotting marginal coverage values
  g = sns.FacetGrid(data_main.
  query("stats == 'mean_coverage'"), col = "data", col_wrap = 3, hue = "methods",
  despine = False, margin_titles = True, legend_out = True,
  sharey = True,
  height = 5)
  g.map(plt.errorbar, "methods", "value", "sd", marker = "o")
  g.map(plt.axhline, y = 0.9, ls='--', c='red')
  g.map(plt.axvline, x = 4.5, color = 'k', linestyle = "dashed")
  
  g.figure.subplots_adjust(wspace=0, hspace=0)
  g.add_legend(bbox_to_anchor = (1.1, 0.5), title = "Methods")
  legend = g.legend
  for text in legend.get_texts():
    if text.get_text() in our_methods:
      text.set_fontweight("bold")
      
  g.set_ylabels("Marginal Coverage")
  g.set_xlabels("Methods")
  g.set_xticklabels(rotation = 45)
  g.set_titles(col_template="{col_name}")
  
  for ax in g.axes.flatten():
    if len(ax.get_xticklabels()) != 0:
      for idx in [0,1,5,6,7]:
        ax.get_xticklabels()[idx].set_fontweight("bold")
        
  plt.tight_layout()
  plt.savefig(f"{images_dir}/{marginal_cover}.pdf", bbox_inches="tight")
  
  # returning data final to plot more general graphs
  plt.close('all')
  return(data_main)

used_data = plot_results(data_names)

# returning table only with smis
smis_data = used_data.query("stats == 'smis'").round(4)
smis_data["value_sd"] = smis_data["value"].map(str) + "\n (" + smis_data["sd"].map(str) + ")"
# pivoting methods and smis
a = smis_data.pivot(index = "data",
columns = "methods",
values = "value_sd")


# rounding and putting sd in parenthesis
a.to_csv("temp")
