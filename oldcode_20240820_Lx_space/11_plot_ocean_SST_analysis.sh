#!/bin/bash

source 00_setup.sh

python3 $src_dir/plot_SST_map_and_analysis.py \
    --input-file $data_dir/hycom_GLBv0.08_572_2017010112_t000.nc \
    --lat-rng 25 50 \
    --lon-rng 160 200 \
    --cutoff-wvlen 1000.0 \
    --output $fig_dir/sst_analysis_20170101.svg \
    --lat-avg-interval 5.0 \
    --ylim -1 5 \
    --no-display

