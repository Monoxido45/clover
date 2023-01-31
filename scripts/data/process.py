import argparse
import zipfile

import pandas as pd
from sklearn.datasets import load_svmlight_file

from utils import get_folder

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help="Dataset name.")
parser.add_argument(
    "--n-samples",
    "-n",
    default=None,
    type=int,
    help="Number of samples for big datasets.",
)
parser.add_argument(
    "--seed", "-s", default=0, help="Seed for resampling dataset. Default is 0."
)
args = parser.parse_args()

# Use chatgpt advices. Pandas can download directly from some places. svmlight is good.


def process(dataset, n_samples=None, seed=0):
    # Based on https://github.com/AIgen/QOOB/blob/master/MATLAB/data/loadBlogData.m
    if dataset == "blog":
        with zipfile.ZipFile(f"data/raw/blog/BlogFeedback.zip", "r") as file:
            df = pd.read_csv(file.open("blogData_train.csv"), header=None)

        X, y = df.iloc[:, :280], df.iloc[:, 280]
        data = pd.DataFrame(X)
        data["target"] = y

    # Based on https://github.com/AIgen/QOOB/blob/master/MATLAB/data/loadProteinData.m
    if dataset == "protein":
        df = pd.read_csv(f"data/raw/protein/CASP.csv")
        X, y = df.iloc[:, 1:], df.iloc[:, 0]
        data = pd.DataFrame(X)
        data["target"] = y

    # Raise an error. Must install xlrd.
    # Based on https://github.com/AIgen/QOOB/blob/master/MATLAB/data/loadConcreteData.m
    if dataset == "concrete":
        df = pd.read_excel(f"data/raw/concrete/Concrete_Data.xls")
        X, y = df.iloc[:, :8], df.iloc[:, 8]
        data = pd.DataFrame(X)
        data["target"] = y

    # Based on https://github.com/AIgen/QOOB/blob/master/MATLAB/data/loadNewsData.m
    if dataset == "news":
        with zipfile.ZipFile(f"data/raw/news/OnlineNewsPopularity.zip", "r") as file:
            df = pd.read_csv(file.open("OnlineNewsPopularity/OnlineNewsPopularity.csv"))

        # First column are urls.
        X, y = df.iloc[:, 1:60], df.iloc[:, 60]
        data = pd.DataFrame(X)
        data["target"] = y

    # Based on https://github.com/AIgen/QOOB/blob/master/MATLAB/data/loadKernelData.m
    if dataset == "kernel":
        with zipfile.ZipFile(f"data/raw/kernel/sgemm_product_dataset.zip", "r") as file:
            df = pd.read_csv(file.open("sgemm_product.csv"))

        # First column are urls.
        X, y = df.iloc[:, :15], df.iloc[:, 15:].mean(axis=1)
        data = pd.DataFrame(X)
        data["target"] = y

    # Based on https://github.com/AIgen/QOOB/blob/master/MATLAB/data/loadSuperconductorData.m
    if dataset == "superconductivity":
        with zipfile.ZipFile(
            f"data/raw/superconductivity/superconduct.zip", "r"
        ) as file:
            df = pd.read_csv(file.open("train.csv"))

        # First column are urls.
        X, y = df.iloc[:, :81], df.iloc[:, 81]
        data = pd.DataFrame(X)
        data["target"] = y

    if dataset == "airfoil":
        df = pd.read_table(f"data/raw/airfoil/airfoil_self_noise.dat", header=None)
        X, y = df.iloc[:, :5], df.iloc[:, 5]
        data = pd.DataFrame(X)
        data["target"] = y

    if dataset == "electric":
        df = pd.read_csv(f"data/raw/electric/Data_for_UCI_named.csv")
        X, y = df.iloc[:, :12], df.iloc[:, 12]
        data = pd.DataFrame(X)
        data["target"] = y

    if dataset == "cycle":
        with zipfile.ZipFile(f"data/raw/cycle/CCPP.zip", "r") as file:
            df = pd.read_excel(file.open("CCPP/Folds5x2_pp.xlsx"))

        # First column are urls.
        X, y = df.iloc[:, :4], df.iloc[:, 4]
        data = pd.DataFrame(X)
        data["target"] = y

    if dataset == "winered":
        df = pd.read_csv(f"data/raw/winered/winequality-red.csv", delimiter=";")
        X, y = df.iloc[:, :11], df.iloc[:, 11]
        data = pd.DataFrame(X)
        data["target"] = y

    if dataset == "winewhite":
        df = pd.read_csv(f"data/raw/winewhite/winequality-white.csv", delimiter=";")
        X, y = df.iloc[:, :11], df.iloc[:, 11]
        data = pd.DataFrame(X)
        data["target"] = y

    return data


DATASET = args.dataset
N_SAMPLES = args.n_samples
SEED = args.seed

data = process(DATASET, N_SAMPLES, SEED)

output_folder = get_folder(f"data/processed")
data.to_csv(f"{output_folder}/{DATASET}.csv", index=False)
print(f"-saved: {output_folder}/{DATASET}.csv")
