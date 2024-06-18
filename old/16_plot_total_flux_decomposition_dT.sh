#!/bin/bash

source 00_setup.sh



input_dirs=""
exp_names=""
parameter=dT
input_dir_root=$data_dir/$target_lab
output_dir=$fig_dir/total_flux_decomposition/$parameter

#for dT in 000 010 020 030 040 050 060 070 080 090 100 ; do
for dT in 000 020 040 060 080 100 ; do
    for Lx in "100" ; do
        for U in "15" ; do
            for _bl_scheme in "MYNN25" ; do
                
                input_dirs="$input_dirs ${input_dir_root}/case_mph-off_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}"
                exp_names="$exp_names $dT"
            done
        done
    done
done



mkdir -p $output_dir
N=2

dhr=24
for i in $( seq 2 2 ); do
#for i in 4 ; do
    
    ((j=j%N)); ((j++==0)) && wait

    hrs_beg=$( printf "%02d" $(( $i * $dhr )) )
    hrs_end=$( printf "%02d" $(( ($i + 1) * $dhr )) )
    output_name="$output_dir/comparison_total_flux_${hrs_beg}-${hrs_end}.png"
    output_name_decomp="$output_dir/decomp_comparison_total_flux_${parameter}_${hrs_beg}-${hrs_end}.png"

    python3 src/plot_comparison_total_flux_more_decomposition.py  \
        --input-dirs $input_dirs  \
        --exp-beg-time "2001-01-01 00:00:00" \
        --time-rng $hrs_beg $hrs_end \
        --parameter $parameter \
        --HFX-rng -15 10        \
        --LH-rng 10 35        \
        --exp-names $exp_names \
        --ref-exp-order 1 \
        --wrfout-data-interval 60 \
        --frames-per-wrfout-file 60 \
        --output $output_name \
        --output-decomp $output_name_decomp \
        --no-display 


done

wait

echo "DONE!"
