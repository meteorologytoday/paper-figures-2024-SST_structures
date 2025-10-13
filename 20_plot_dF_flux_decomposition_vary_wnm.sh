#!/bin/bash

source 00_setup.sh



exp_names=""
input_dir_root=$data_dir/$target_lab


for target_lab in lab_SIMPLE lab_FULL ; do
for Ug in 10 20 ; do
for dSST in 050 100 ; do
for bl_scheme in MYNN25 MYJ YSU; do
#for bl_scheme in MYNN25 ; do

    dhr=$( get_dhr $bl_scheme ) 
    hr_beg=120
    hr_end=$(( $hr_beg + $dhr ))

    hr=${hr_beg}-${hr_end}

    gendata_dir=$( gen_gendata_dir $Ug )
    input_file=$gendata_dir/dF_phase_analysis/fixed_dSST/$target_lab/collected_flux_${bl_scheme}_hr${hr}.nc
    output_dir=$fig_dir/dF_flux_decomposition_varying_wnm/$target_lab
    output_file=$output_dir/dF_flux_decomposition_onefig_U${Ug}_dSST${dSST}_varying_wnm_${bl_scheme}_hr${hr}.svg

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
        --thumbnail-numbering abcdefg \
        --domain-size $Lx \
        --no-display
            
done
done
done
done


