#!/bin/bash

nproc=1

archive_root=/home/t2hsu/projects/SST-front/11_SST_gaussian/lab_gaussian
output_root=./output_budget_analysis
casename=case_mph-off_dT100_wid050_woML_MYNN3_wpkt01

input_dir=$archive_root/$casename
output_dir=$output_root/$casename

beg_hr=12
end_hr=36

python3 budget_analysis.py  \
    --input-dir $input_dir  \
    --output-dir $output_dir \
    --time-rng $(( $beg_hr * 60 ))  $(( $end_hr * 60 )) \
    --wrfout-data-interval 60 \
    --frames-per-wrfout-file 60 \
    --nproc $nproc \
    --exp-beg-time 2001-01-01


