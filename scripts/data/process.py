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
        with zipfile.ZipFile(f"data/raw/{dataset}/BlogFeedback.zip", "r") as file:
            df = pd.read_csv(file.open("blogData_train.csv"))

        X, y = df.iloc[:, :280], df.iloc[:, 281]
        data = pd.DataFrame(X)
        data["target"] = y

    # Based on https://github.com/AIgen/QOOB/blob/master/MATLAB/data/loadProteinData.m
    if dataset == "protein":
        df = pd.read_csv(f"data/raw/protein/CASP.csv")
        X, y = df.iloc[:, 1:], df.iloc[:, 0]
        data = pd.DataFrame(X)
        data["target"] = y

    # Raise an error. Must install xlrd.
    if dataset == "concrete":
        df = pd.read_xml(f"data/raw/concrete/Concrete_Data.xls")
        X, y = df.iloc[:, :8], df.iloc[:, 8]
        data = pd.DataFrame(X)
        data["target"] = y

    if dataset == "news":
        with zipfile.ZipFile(f"data/raw/news/OnlineNewsPopularity.zip", "r") as file:
            df = pd.read_csv(file.open("OnlineNewsPopularity/OnlineNewsPopularity.csv"))

        # First column are urls.
        X, y = df.iloc[:, 1:60], df.iloc[:, 60]
        data = pd.DataFrame(X)
        data["target"] = y
        pass

    if dataset == "kernel":
        pass

    if dataset == "superconductivity":
        pass

    if dataset == "airfoil":
        pass

    if dataset == "electric":
        pass

    if dataset == "cycle":
        pass

    if dataset == "winered":
        pass

    if dataset == "winewhite":
        pass

    return data


DATASET = args.dataset
N_SAMPLES = args.n_samples
SEED = args.seed

data = process(DATASET, N_SAMPLES, SEED)

output_folder = get_folder(f"data/processed")
data.to_csv(f"{output_folder}/{DATASET}.csv", index=False)
print(f"-saved: {output_folder}/{DATASET}.csv")
