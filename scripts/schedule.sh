#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py --config-name=train_edm

python src/train.py --config-name=train_consistencyCT
