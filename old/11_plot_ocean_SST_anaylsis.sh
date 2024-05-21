#!/bin/bash

source 00_setup.sh

python3 $src_dir/plot_SST_map_and_analysis.py \
    --input-file $data_dir/hycom_GLBv0.08_572_2017010112_t000.nc \
    --lat-rng 25 50 \
    --lon-rng -180 -140 \
    --cutoff-wvlen 2.0 \
    --output-SSTmap $fig_dir/sst_analysis_map.png \
    --output-SSTspec $fig_dir/sst_analysis_spec.png \
    --no-display


