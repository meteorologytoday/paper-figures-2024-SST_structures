#!/bin/bash

source 00_setup.sh



exp_names=""
parameter=dT
output_dir=$fig_dir/dF_total_flux_decomposition/$parameter


for target_lab in lab_FIXEDDOMAIN_SST_sine_WETLWSW ; do
for Ug in 20 ; do
for wnm in 004 ; do
#for bl_scheme in MYNN25 YSU MYJ; do
for bl_scheme in MYNN25 ; do
for hr in 120-240 ; do

    input_file=$gendata_dir/dF_collected_flux/fixed_wnm/$target_lab/collected_flux_${bl_scheme}_hr${hr}.nc
    output_dir=$fig_dir/dF_flux_decomposition_varying_dSST/$target_lab
    output_file=$output_dir/dF_flux_decomposition_onefig_varying_dSST_${bl_scheme}_hr${hr}.svg

    mkdir -p $output_dir

    python3 src/plot_flux_decomp.py \
        --input-file $input_file \
        --output $output_file \
        --delta-analysis \
        --varying-param dSST \
        --fixed-params Ug wnm \
        --fixed-param-values $Ug $wnm \
        --LH-rng -35 10 \
        --HFX-rng -5 5 \
        --spacing 0.02 \
        --no-display
            
done
done
done
done
done


