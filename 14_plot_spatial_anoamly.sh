#!/bin/bash


source 00_setup.sh

output_dir=$fig_dir/spatial_anomaly
input_dirs=""
exp_names=""
input_dir_root=$data_dir/$target_lab



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
#for i in $(seq 0 1 2 ); do
for i in $(seq 1 2 ) 0; do
    
    ((j=j%N)); ((j++==0)) && wait

    hrs_beg=$( printf "%02d" $(( $i * $dhr )) )
    hrs_end=$( printf "%02d" $(( ($i + 1) * $dhr )) )
    output_name="$output_dir/spatial_anomaly_${hrs_beg}-${hrs_end}.png"

    python3 src/plot_anomaly.py \
        --input-dirs $input_dirs  \
        --exp-beg-time "2001-01-01 00:00:00" \
        --time-rng $hrs_beg $hrs_end \
        --x-rng 0 200        \
        --exp-names $exp_names \
        --ref-exp-order 1 \
        --wrfout-data-interval 60 \
        --frames-per-wrfout-file 60 \
        --output $output_name \
        --no-display 

done

wait

echo "DONE!"
