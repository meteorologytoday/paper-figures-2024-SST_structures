#!/bin/bash


source 00_setup.sh

selected_dSST=1.0
RH=0.999

output_dir=$fig_dir/phase_SQ15
SQ15_file=$gendata_dir/phase_SQ15_E0_4.80e-06.nc
#SQ15_file=$gendata_dir/phase_SQ15.nc
mkdir -p $output_dir

for Ug in 20 ; do
for RH in 0.999 ; do
for DTheta in 0.1 1 5 10 ; do

    python3 src/plot_param_space_SQ15_new.py  \
        --input-file $SQ15_file \
        --varying-params Lx dSST \
        --param1-rng 0 200 \
        --param2-rng 0 1 \
        --fixed-params Ug RH DTheta \
        --fixed-param-values $Ug $RH $DTheta \
        --output $output_dir/phase_diagram-Ug_${Ug}-DTheta_${DTheta}-RH_${RH}.svg \
        --no-display

done
done
done
