#!/bin/bash

source 00_setup.sh

exp_names=""
input_dir_root=$data_dir/$target_lab


for target_lab in lab_FULL ; do
for Ug in ${Us[@]} ; do
for wnm in 010 ; do
#for hr in 120-240 ; do
for hr in 240-360 ; do

    input_files=""
    labels=""
    for bl_scheme in MYNN25 MYJ YSU; do
        input_file=$gendata_dir/dF_phase_analysis/fixed_wnm/$target_lab/collected_flux_${bl_scheme}_hr${hr}.nc
        input_files="$input_files $input_file"
        labels="$labels $bl_scheme"
    done

    colors="black orangered dodgerblue"

    output_dir=$fig_dir/phase_misc/$target_lab
    output_file=$output_dir/phase_misc_wnm${wnm}_varying_dSST_hr${hr}.svg

    mkdir -p $output_dir

    python3 src/plot_response_to_SST_multiple.py \
        --input-files $input_files \
        --output $output_file \
        --colors $colors \
        --varnames IWV PRECIP  \
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

