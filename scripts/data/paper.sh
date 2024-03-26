#!/bin/bash

# Download datasets
python3 scripts/data/download.py -d blog
python3 scripts/data/download.py -d protein
python3 scripts/data/download.py -d concrete
python3 scripts/data/download.py -d news
python3 scripts/data/download.py -d kernel
python3 scripts/data/download.py -d superconductivity
python3 scripts/data/download.py -d airfoil
python3 scripts/data/download.py -d electric
python3 scripts/data/download.py -d cycle
python3 scripts/data/download.py -d winered
python3 scripts/data/download.py -d winewhite

# Process datasets
python3 scripts/data/process.py -d blog
python3 scripts/data/process.py -d protein
python3 scripts/data/process.py -d concrete
python3 scripts/data/process.py -d news
python3 scripts/data/process.py -d kernel
python3 scripts/data/process.py -d superconductivity
python3 scripts/data/process.py -d airfoil
python3 scripts/data/process.py -d electric
python3 scripts/data/process.py -d cycle
python3 scripts/data/process.py -d winered
python3 scripts/data/process.py -d winewhite
python3 scripts/data/process.py -d bike
python3 scripts/data/process.py -d meps19
python3 scripts/data/process.py -d star
