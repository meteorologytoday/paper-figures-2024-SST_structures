#!/bin/bash

source 00_setup.sh



input_file=$gendata_dir/gendata/collected_flux/lab_sine_wetlwsw/collected_flux_MYNN25_hr240-360.nc
exp_names=""
parameter=dT
input_dir_root=$data_dir/$target_lab
output_dir=$fig_dir/total_flux_decomposition/$parameter


for target_lab in lab_sine_wetlwsw ; do
for Ug in 20 ; do
for dSST in 300 ; do
for bl_scheme in MYNN25 ; do
for hr in 240-360 ; do

    input_file=$gendata_dir/collected_flux/$target_lab/collected_flux_${bl_scheme}_hr${hr}.nc
    output_dir=$fig_dir/flux_decomposition_varying_Lx/$target_lab
    output_file=$output_dir/flux_decomposition_varying_Lx_${bl_scheme}_hr${hr}.svg

    mkdir -p $output_dir

    python3 src/plot_comparison_total_flux_new.py \
        --input-file $input_file \
        --output $output_file \
        --varnames      HFX     C_H_WND_TOA  WND_TOA_cx_mul_C_H  C_H_TOA_cx_mul_WND  C_H_WND_cx_mul_TOA  \
                        LH      C_Q_WND_QOA  WND_QOA_cx_mul_C_Q  C_Q_QOA_cx_mul_WND  C_Q_WND_cx_mul_QOA  \
        --varying-param Lx \
        --fixed-params Ug dSST \
        --fixed-param-values $Ug $dSST \
        --ncols 5 \
        --no-display
            
done
done
done
done
done


