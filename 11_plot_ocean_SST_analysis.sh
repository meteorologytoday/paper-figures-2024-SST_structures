#!/bin/bash

source 00_setup.sh

python3 $src_dir/plot_SST_map_and_analysis.py \
    --input-file $data_dir/hycom_glby_930_2024010112_t000_t_depth5.nc \
    --lat-rng 25 50 \
    --lon-rng 160 200 \
    --cutoff-wvlen 1000.0 \
    --output $fig_dir/sst_analysis_20240101.svg \
    --lat-avg-interval 5.0 \
    --ylim -5 1 \
    --no-display



