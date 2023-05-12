#!/bin/bash

# Download datasets
python scripts/data/download.py -d blog
python scripts/data/download.py -d protein
python scripts/data/download.py -d concrete
python scripts/data/download.py -d news
python scripts/data/download.py -d kernel
python scripts/data/download.py -d superconductivity
python scripts/data/download.py -d airfoil
python scripts/data/download.py -d electric
python scripts/data/download.py -d cycle
python scripts/data/download.py -d winered
python scripts/data/download.py -d winewhite

# Process datasets
python scripts/data/process.py -d blog
python scripts/data/process.py -d protein
python scripts/data/process.py -d concrete
python scripts/data/process.py -d news
python scripts/data/process.py -d kernel
python scripts/data/process.py -d superconductivity
python scripts/data/process.py -d airfoil
python scripts/data/process.py -d electric
python scripts/data/process.py -d cycle
python scripts/data/process.py -d winered
python scripts/data/process.py -d winewhite
python scripts/data/process.py -d bike
python scripts/data/process.py -d meps19
python scripts/data/process.py -d star
