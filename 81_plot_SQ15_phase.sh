#!/bin/bash


source 00_setup.sh

selected_dSST=1.0
RH=0.999

output_dir=$fig_dir/phase_SQ15

mkdir -p $output_dir

for DTheta in 0.1 1 5 10 ; do

    python3 src/plot_param_space.py  \
        --input-dir $data_SQ15_dir/output_strong_E0 \
        --selected-dSST $selected_dSST \
        --Ugs 0 5 10 15 20 \
        --wvlens 20 40 60 80 100 120 140 160 180 200 \
        --selected-DTheta $DTheta \
        --output $output_dir/phase_diagram-DTheta_${DTheta}-RH_${RH}.png \
        --RH ${RH} \
        --no-display

done
