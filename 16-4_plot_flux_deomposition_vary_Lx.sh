#!/bin/bash

source 00_setup.sh



exp_names=""
parameter=dT
input_dir_root=$data_dir/$target_lab
output_dir=$fig_dir/total_flux_decomposition/$parameter


for target_lab in lab_sine_WETLWSW ; do
for Ug in 20 ; do
for dSST in 300 ; do
for bl_scheme in MYNN25 MYJ YSU; do
for hr in 240-360 ; do


    input_file=$gendata_dir/collected_flux/fixed_dSST/$target_lab/collected_flux_${bl_scheme}_hr${hr}.nc
    output_dir=$fig_dir/flux_decomposition_varying_Lx/$target_lab
    output_file=$output_dir/flux_decomposition_varying_Lx_${bl_scheme}_hr${hr}.svg

    mkdir -p $output_dir

    python3 src/plot_comparison_total_flux_new_one_fig.py \
        --input-file $input_file \
        --output $output_file \
        --varying-param Lx \
        --fixed-params Ug dSST \
        --fixed-param-values $Ug $dSST \
        --LH-rng  -5 15 \
        --HFX-rng -5 10 \
        --spacing 0.05 \
        --no-display
            
done
done
done
done
done


