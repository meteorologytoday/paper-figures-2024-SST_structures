#!/bin/bash

source 00_setup.sh

$py $src_dir/plot_AR_freq_with_std.py \
    --input $diagdata_dir/AR_interannual_statistics_ANOM_LEN_${yrng_str}.nc \
    --output $fig_dir/AR_freq_std.png \
    --no-display

$py $src_dir/plot_EOF_analysis.py \
    --input $diagdata_dir/EOF.nc --input-NINO $climidx_dir/NINO34.nc \
    --input-PDO $climidx_dir/PDO.nc --output-EOF $fig_dir/AR_EOF.png \
    --output-timeseries $fig_dir/AR_EOF_timeseries.png \
    --no-display

