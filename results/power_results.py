# plotting power simulation results for RF coverage evaluator
# function that reads all npy files
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


plt.style.use('ggplot')
sns.set_palette("Set1")
path_original = os.getcwd()
type_reg = "gb_quantile"

def import_all_results(type_reg, ntrain = np.arange(1000, 25000, 5000), coverage_evaluators = "RF", d = 50, kind = "bimodal", color = "red", images_dir = "results/figures", figname = "power_experiment.pdf"):
  # checking if there is more than one coverage evaluator
  if not isinstance(coverage_evaluators, dict):
    power_tests = np.zeros(ntrain.shape[0])
    for i in range(ntrain.shape[0]):
      power_tests[i] = np.load("results/pickle_files/power_tests_{}_n_{}_d_{}_{}_{}_data.npy".format(
        type_reg, ntrain[i], d, coverage_evaluators, kind))
    # plotting and saving
    plt.figure(figsize = (8, 6))
    plt.plot(ntrain, power_tests, color = color, label  = coverage_evaluators)
    plt.xlabel("Number of training  samples")
    plt.ylabel("Test power")
    plt.legend(loc=1)
    plt.title("Test power experiment for " + kind + " data and " + coverage_evaluators + " coverage evaluator")
    plt.show()
    plt.savefig(f"{images_dir}/{figname}")
  elif isinstance(coverage_evaluators, dict):
    # bundling all power statistics into a matrix
    power_tests =  np.zeros((ntrain.shape[0], len(coverage_evaluators)))
    keys = list(coverage_evaluators.keys())
    
    for i in range(ntrain.shape[0]):
      for j in range(len(coverage_evaluators)):
        power_tests[i, j] = np.load("results/pickle_files/power_tests_{}_n_{}_d_{}_{}_{}_data.npy".format(
        type_reg, ntrain[i], d, coverage_evaluators[keys[j]], kind))
    
    # plotting all coverage evaluator power   
    cols = sns.color_palette("Set1")
    plt.figure(figsize = (8, 6))
    for i in range(len(coverage_evaluators)):
      plt.plot(ntrain, power_tests[:, i], color = cols[i], label  = keys[i])
    plt.xlabel("Number of training  samples")
    plt.ylabel("Test power")
    plt.legend(loc= (0.85, 0))
    plt.title("Test power experiment for " + kind + " data")
    plt.show()
    plt.savefig(f"{images_dir}/{figname}")

# running script for RF and logistic regression
ce_dict = {"RF":"RF",
          "Logistic":"LogisticRegression(penalty='none')"}

# importing and plotting all results
import_all_results(type_reg = "gb_quantile", coverage_evaluators = ce_dict)
