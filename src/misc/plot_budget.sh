#!/bin/bash

nproc=1


casename=case_mph-off_dT100_wid050_woML_MYNN3_wpkt01
input_dir=./output_budget_analysis/$casename
output_dir=./output_figure_budget_analysis/$casename

beg_hr=18
beg_min=0
end_hr=24
end_min=0



for z_idx in 0 1 2 3 4 5; do

    python3 plot_budget_analaysis.py  \
        --input-dir $input_dir  \
        --output-dir $output_dir \
        --time-rng $(( $beg_hr * 60 + $beg_min ))  $(( $end_hr * 60 + $end_min )) \
        --wrfout-data-interval 60 \
        --frames-per-wrfout-file 1 \
        --nproc $nproc \
        --overlay \
        --z-idx  $z_idx \
        --exp-beg-time 2001-01-01

done
