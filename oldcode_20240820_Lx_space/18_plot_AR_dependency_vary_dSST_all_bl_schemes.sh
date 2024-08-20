#!/bin/bash

source 00_setup.sh

exp_names=""
input_dir_root=$data_dir/$target_lab


for target_lab in lab_sine_WETLWSW ; do
for Ug in 20 ; do
for Lx in 500 ; do
for hr in 240-360 120-240 ; do

    input_files=""
    labels=""
    for bl_scheme in MYNN25 MYJ YSU; do
        input_file=$gendata_dir/collected_flux/fixed_Lx/$target_lab/collected_flux_${bl_scheme}_hr${hr}.nc
        input_files="$input_files $input_file"
        labels="$labels $bl_scheme"
    done

    colors="black orangered dodgerblue"

    output_dir=$fig_dir/AR_dependency/$target_lab
    #output_file=$output_dir/AR_dependency_varying_dSST_${bl_scheme}_hr${hr}.svg
    output_file=$output_dir/AR_dependency_varying_dSST_hr${hr}.svg

    mkdir -p $output_dir

    python3 src/plot_comparison_total_flux_new.py \
        --input-files $input_files \
        --output $output_file \
        --colors $colors \
        --varnames IVT IWV PRECIP  \
        --varying-param dSST  \
        --fixed-params Ug Lx  \
        --fixed-param-values $Ug $Lx \
        --spacing 0.02 \
        --labels $labels \
        --no-display
            
done
done
done
done

