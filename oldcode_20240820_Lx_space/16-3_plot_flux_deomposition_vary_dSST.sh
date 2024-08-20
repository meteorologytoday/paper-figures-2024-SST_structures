#!/bin/bash

source 00_setup.sh



exp_names=""
parameter=dT
input_dir_root=$data_dir/$target_lab
output_dir=$fig_dir/total_flux_decomposition/$parameter


for target_lab in lab_sine_DRY lab_sine_WETLWSW ; do
for Ug in 20 ; do
for Lx in 500 ; do
#for bl_scheme in MYNN25 YSU MYJ; do
for bl_scheme in MYNN25 ; do
for hr in 120-240 ; do

    input_file=$gendata_dir/collected_flux/fixed_Lx/$target_lab/collected_flux_${bl_scheme}_hr${hr}.nc
    output_dir=$fig_dir/flux_decomposition_varying_dSST/$target_lab
    output_file=$output_dir/flux_decomposition_onefig_varying_dSST_${bl_scheme}_hr${hr}.svg

    mkdir -p $output_dir

    python3 src/plot_comparison_total_flux_new_one_fig.py \
        --input-file $input_file \
        --output $output_file \
        --varying-param dSST \
        --fixed-params Ug Lx \
        --fixed-param-values $Ug $Lx \
        --LH-rng -35 10 \
        --HFX-rng -5 5 \
        --spacing 0.02 \
        --no-display
            
done
done
done
done
done


