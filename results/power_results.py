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

def import_all_results(type_reg, ntrain = np.arange(1000, 25000, 5000), coverage_evaluator = "RF", d = 50, kind = "bimodal", color = "red", images_dir = "results/figures", figname = "power_experiment.pdf"):
  power_tests = np.zeros(ntrain.shape[0])
  for i in in range(ntrain.shape[0]):
    power_tests[i] = np.load("results/pickle_files/power_tests_{}_n_{}_d_{}_{}_data.npy".format(
      type_reg, ntrain[i], d, coverage_evaluator, kind))
  
  # plotting and saving
  plt.figure(figsize = (8, 6))
  plt.plot(ntrain, power_tests, color = color, label  = coverage_evaluator)
  plt.xlabel("Number of training  samples")
  plt.ylabel("Test power")
  plt.legend(loc=1)
  plt.title("Test power experiment for " + kind + " data and " + coverage_evaluator + " coverage evaluator")
  plt.show()
  plt.savefig(f"{images_dir}/{figname}")
    
    
# running the script
if __name__ == '__main__':
  import_all_results(type_reg = type_reg)
