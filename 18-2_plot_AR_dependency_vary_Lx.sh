#!/bin/bash

source 00_setup.sh

exp_names=""
input_dir_root=$data_dir/$target_lab


for target_lab in lab_sine_DRY lab_sine_WETLWSW ; do
for Ug in 20 ; do
for dSST in 300 ; do
#for bl_scheme in MYNN25 YSU MYJ; do
for bl_scheme in MYNN25 ; do
for hr in 120-240 240-360 ; do

    input_files=""
    labels=""
    for bl_scheme in MYNN25 MYJ YSU; do
        input_file=$gendata_dir/collected_flux/fixed_dSST/$target_lab/collected_flux_${bl_scheme}_hr${hr}.nc
        input_files="$input_files $input_file"
        labels="$labels $bl_scheme"
    done

    colors="black orangered dodgerblue"

    output_dir=$fig_dir/AR_dependency/$target_lab
    output_file=$output_dir/AR_dependency_varying_Lx_hr${hr}.svg

    mkdir -p $output_dir

    python3 src/plot_comparison_total_flux_new.py \
        --input-files $input_files \
        --output $output_file \
        --varnames IVT IWV PRECIP   \
        --colors $colors \
        --varying-param Lx  \
        --fixed-params Ug dSST  \
        --fixed-param-values $Ug $dSST \
        --thumbnail-numbering defghijklmn \
        --spacing 5 \
        --labels $labels \
        --no-display
            
done
done
done
done
done

