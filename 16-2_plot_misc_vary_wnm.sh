#!/bin/bash

source 00_setup.sh

exp_names=""
input_dir_root=$data_dir/$target_lab


for target_lab in lab_FULL ; do
for Ug in 20 ; do
for dSST in 100 ; do
for hr in 120-240  ; do

    input_files=""
    labels=""
    for bl_scheme in MYNN25 MYJ YSU ; do
        input_file=$gendata_dir/dF_phase_analysis/fixed_dSST/$target_lab/collected_flux_${bl_scheme}_hr${hr}.nc
        input_files="$input_files $input_file"
        labels="$labels $bl_scheme"
    done

    colors="black orangered dodgerblue"

    output_dir=$fig_dir/phase_misc/$target_lab
    output_file=$output_dir/phase_misc_dSST${dSST}_varying_wnm_hr${hr}.svg

    mkdir -p $output_dir

    python3 src/plot_response_to_SST_multiple.py \
        --input-files $input_files \
        --output $output_file \
        --varnames IWV PRECIP   \
        --colors $colors \
        --varying-param wnm  \
        --fixed-params Ug dSST  \
        --fixed-param-values $Ug $dSST \
        --thumbnail-numbering cdefghijklmn \
        --spacing 5 \
        --labels $labels \
        --domain-size $Lx \
        --no-display
            
done
done
done
done

