#!/bin/bash

fig_dir=figures/spatial_anomaly

input_dirs=""
exp_names=""
input_dir_root=data/runs


for dT in 000 010 020 030 040 050 060 070 080 090 100 ; do
    for Lx in "100" ; do
        for U in "15" ; do
            for _bl_scheme in "MYNN25" ; do
                
                input_dirs="$input_dirs ${input_dir_root}/case_mph-off_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}"
                exp_names="$exp_names $dT"
            done
        done
    done
done

output_dir=$fig_dir
mkdir -p $output_dir
N=2

dhr=6
for i in $(seq 7 12 ); do
    
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
