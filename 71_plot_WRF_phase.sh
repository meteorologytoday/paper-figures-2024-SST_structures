#!/bin/bash

source 00_setup.sh

flux_dir=$gendata_dir/flux_decomposition
dhr=24
bl_scheme="MYNN25"

                
for day in 1; do
    
    hrs_beg=$( printf "%02d" $(( $day * $dhr )) )
    hrs_end=$( printf "%02d" $(( ($day + 1) * $dhr )) )
   
    input_dir="${flux_dir}/hr${hrs_beg}-${hrs_end}"
    input_file="${input_dir}/case_mph-off_${bl_scheme}.nc"

    output_file=${fig_dir}/param_space_WRF_hr${hrs_beg}-${hrs_end}.png

    eval "python3 src/plot_param_space_WRF.py  \
        --input $input_file  \
        --output ${output_file} \
        --Ug-rng 0 20 \
        --selected-dSST 100 \
        --title 'Time: ${hrs_beg}-${hrs_end} hr' \
        --no-display
    "
done

echo "Done"
