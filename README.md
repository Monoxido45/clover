# Local Calibration Validation
Perfom local calibration validation by testing $H_0: \mathbb{P}(Y \in C(X)|X) = 1 - \alpha$

## Instructions

All scripts must be run from the root directory (`/lcv`).

### Downloading datasets

To download and process all datasets, run `bash scripts/data/paper.sh`. 

If you want to download a single dataset, run `python scripts/data/download.py -d $DATASET` where `$DATASET` is any of the datasets listed as keys in the `URL` dictionary in `scripts/data/download.py`. To process this dataset, run `python scripts/data/process.py -d $DATASET`. 


