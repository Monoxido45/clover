import argparse

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
    if dataset in ["cpusmall"]:
        X, y = load_svmlight_file(f"data/raw/{dataset}/{dataset}")
        data = pd.DataFrame(X.toarray())
        data["target"] = pd.DataFrame(y)

    return data


DATASET = args.dataset
N_SAMPLES = args.n_samples
SEED = args.seed

data = process(DATASET, N_SAMPLES, SEED)

output_folder = get_folder(f"data/processed")
data.to_csv(f"{output_folder}/{DATASET}.csv", index=False)
print(f"-saved: {output_folder}/{DATASET}.csv")
