#!/bin/bash

fig_dir=figures/spatial_anomaly

input_dirs=""
exp_names=""
input_dir_root=data/runs/lab_sine_200km
for dT in 000 010 020 030 040 050 060 070 080 090 100 ; do
#for dT in 000 020 040 ; do
        for MLsetup in woML ; do
            for MLscheme in MYNN25 ; do 
                for wnm in 01 ; do
                    input_dirs="$input_dirs $input_dir_root/case_mph-off_dT${dT}_wnm${wnm}_${MLsetup}_${MLscheme}"
                    exp_names="$exp_names $dT"
                done
            done
    done
done

output_dir=$fig_dir
mkdir -p $output_dir
N=2

dhr=6
for i in 4; do
    
    ((j=j%N)); ((j++==0)) && wait

    hrs_beg=$( printf "%02d" $(( $i * $dhr )) )
    hrs_end=$( printf "%02d" $(( ($i + 1) * $dhr )) )
    output_name="$output_dir/spatial_anomaly_${hrs_beg}-${hrs_end}.png"

    python3 src/plot_anomaly.py \
        --input-dirs $input_dirs  \
        --exp-beg-time "2001-01-01 00:00:00" \
        --time-rng $hrs_beg $hrs_end \
        --x-rng 0 200        \
        --HFX-rng -15 10        \
        --LH-rng 10 35        \
        --exp-names $exp_names \
        --ref-exp-order 1 \
        --wrfout-data-interval 60 \
        --frames-per-wrfout-file 60 \
        --output $output_name \
        --no-display 

done

wait

echo "DONE!"
