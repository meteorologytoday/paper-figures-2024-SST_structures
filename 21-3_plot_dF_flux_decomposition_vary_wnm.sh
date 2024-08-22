#!/bin/bash

source 00_setup.sh



exp_names=""
input_dir_root=$data_dir/$target_lab


for target_lab in lab_FIXEDDOMAIN_SST_sine_WETLWSW ; do
for Ug in 20 ; do
for dSST in 300 ; do
#for bl_scheme in MYNN25 MYJ YSU; do
for bl_scheme in MYNN25 ; do
for hr in 120-240 ; do


    input_file=$gendata_dir/dF_collected_flux/fixed_dSST/$target_lab/collected_flux_${bl_scheme}_hr${hr}.nc
    output_dir=$fig_dir/dF_flux_decomposition_varying_wnm/$target_lab
    output_file=$output_dir/dF_flux_decomposition_onefig_varying_wnm_${bl_scheme}_hr${hr}.svg

    mkdir -p $output_dir

    python3 src/plot_flux_decomp.py \
        --input-file $input_file \
        --output $output_file \
        --delta-analysis \
        --varying-param wnm \
        --fixed-params Ug dSST \
        --fixed-param-values $Ug $dSST \
        --LH-rng  -5 15 \
        --HFX-rng -5 10 \
        --spacing 4.0 \
        --thumbnail-numbering cdefg \
        --no-display
            
done
done
done
done
done

