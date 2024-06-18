#!/bin/bash


source 00_setup.sh

selected_dSST=1.0
RH=0.999

output_dir=$fig_dir/phase_SQ15

mkdir -p $output_dir

for DTheta in 0.1 1 5 10 ; do

    python3 src/plot_param_space_SQ15_new.py  \
        --input-dir $data_SQ15_dir/output_strong_E0 \
        --varying-params Lx dSST \
        --param1-rng 0 200 \
        --param2-rng 0 1 \
        --fixed-params Ug \
        --fixed-param-values $Ug \
        --selected-DTheta $DTheta \
        --output $output_dir/phase_diagram-DTheta_${DTheta}-RH_${RH}.png \
        --RH ${RH} \
        --no-display

done
