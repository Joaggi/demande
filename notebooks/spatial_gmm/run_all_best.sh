#!/bin/bash

python3 dmkde_best.py dmkde spatial_gmm
python3 dmkde_sgd_best.py dmkde_sgd spatial_gmm
python3 inverse_maf_best.py inverse_maf spatial_gmm
python3 made_best.py made spatial_gmm
python3 neural_splines_best.py neural_splines spatial_gmm
python3 planar_flow_best.py planar_flow spatial_gmm
