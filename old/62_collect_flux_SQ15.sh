#!/bin/bash


source 00_setup.sh

#output=$gendata_dir/phase_SQ15_E0_4.80e-06.nc
output=$gendata_dir/phase_SQ15_E0_6.00e-07.nc



#input_dir=$data_SQ15_dir/output_strong_E0
#input_dir=$data_SQ15_dir/output-E0_4.80e-06
input_dir=$data_SQ15_dir/output-E0_6.00e-07

    #--dSSTs 0.0 0.2 0.4 0.6 0.8 1.0                 \
    #--DThetas 0.1 1 5 10                            \
python3 src/collect_flux_analysis_SQ15.py           \
    --input-dir $input_dir \
    --dSSTs 0.0 0.2 0.4 0.6 0.8 1.0                 \
    --Ugs   0 5 10 15 20 25 30                      \
    --wvlens 20 40 60 80 100 120 140 160 180 200    \
    --DThetas 3 4 5                                 \
    --RHs 0.999                                     \
    --output $output


