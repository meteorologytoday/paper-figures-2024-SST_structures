#!/bin/bash


source 00_setup.sh

output=$gendata_dir/phase_SQ15.nc

    #--dSSTs 0 0.2 0.4 0.6 0.8 1.0                   \
python3 src/collect_flux_analysis_SQ15.py           \
    --input-dir $data_SQ15_dir/output_strong_E0     \
    --dSSTs 1.0                   \
    --Ugs   0 5 10 15 20                            \
    --wvlens 20 40 60 80 100 120 140 160 180 200    \
    --DThetas 0.1 1 5 10                            \
    --RHs 0.999                                     \
    --output $output


