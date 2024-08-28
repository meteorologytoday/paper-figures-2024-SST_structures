#!/bin/bash

source 00_setup.sh

exp_names=""
input_dir_root=$data_dir/$target_lab


for target_lab in lab_FIXEDDOMAIN_SST_sine_WETLWSW ; do
for Ug in 20 ; do
for wnm in 004 ; do
for hr in 0-72 0-120 120-240 ; do

    input_files=""
    labels=""
    #for bl_scheme in MYNN25 MYJ YSU; do
    for bl_scheme in MYNN25 ; do
        input_file=$gendata_dir/dF_phase_analysis/fixed_wnm/$target_lab/collected_flux_${bl_scheme}_hr${hr}.nc
        input_files="$input_files $input_file"
        labels="$labels $bl_scheme"
    done

    colors="black orangered dodgerblue"

    output_dir=$fig_dir/AR_dependency/$target_lab
    #output_file=$output_dir/AR_dependency_varying_dSST_${bl_scheme}_hr${hr}.svg
    output_file=$output_dir/AR_dependency_wnm${wnm}_varying_dSST_hr${hr}.svg

    mkdir -p $output_dir

    python3 src/plot_response_to_SST_multiple.py \
        --input-files $input_files \
        --output $output_file \
        --colors $colors \
        --varnames IVT IWV PRECIP  \
        --varying-param dSST  \
        --fixed-params Ug wnm  \
        --fixed-param-values $Ug $wnm \
        --spacing 0.02 \
        --labels $labels \
        --no-display
            
done
done
done
done

