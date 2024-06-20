import argparse

from utils import get_folder, simple_download_from_url, download_from_url

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d")
args = parser.parse_args()

DATASET = args.dataset

URL = {
    "blog": "http://archive.ics.uci.edu/ml/machine-learning-databases/00304/",
    "protein": "http://archive.ics.uci.edu/ml/machine-learning-databases/00265/",
    "concrete": "http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/",
    "news": "http://archive.ics.uci.edu/ml/machine-learning-databases/00332/",
    "kernel": "http://archive.ics.uci.edu/ml/machine-learning-databases/00440/",
    "superconductivity": "http://archive.ics.uci.edu/ml/machine-learning-databases/00464/",
    "airfoil": "http://archive.ics.uci.edu/ml/machine-learning-databases/00291/",
    "electric": "http://archive.ics.uci.edu/ml/machine-learning-databases/00471/",
    "cycle": "http://archive.ics.uci.edu/ml/machine-learning-databases/00294/",
    "winered": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/",
    "winewhite": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/",
    "amazon": "https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews",
    "SGEMM": "http://archive.ics.uci.edu/dataset/440/sgemm+gpu+kernel+performance",
    "yearprediction": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#YearPredictionMSD",
    "WEC": "http://archive.ics.uci.edu/dataset/882/large-scale+wave+energy+farm",
}
FILES = {
    "blog": ["BlogFeedback.zip"],
    "protein": ["CASP.csv"],
    "concrete": ["Concrete_Data.xls", "Concrete_Readme.txt"],
    "news": ["OnlineNewsPopularity.zip"],
    "kernel": ["sgemm_product_dataset.zip"],
    "superconductivity": ["superconduct.zip"],
    "airfoil": ["airfoil_self_noise.dat"],
    "electric": ["Data_for_UCI_named.csv"],
    "cycle": ["CCPP.zip"],
    "winered": ["winequality-red.csv", "winequality.names"],
    "winewhite": ["winequality-white.csv", "winequality.names"],
    "amazon": [""],
    "SGEMM": [""],
    "yearprediction": [""],
    "WEC": [""],
}

output_folder = get_folder(f"data/raw/{DATASET}")

for file in FILES[DATASET]:
    file_url = f"{URL[DATASET]}/{file}"
    file_path = f"{output_folder}/{file}"

    if file == "":
        raise ValueError("Cannot be downloaded. Check repo for specific instructions.")

    try:
        download_from_url(file_url, file_path)
    except KeyError:
        simple_download_from_url(file_url, file_path)
