#!/bin/bash

source 00_setup.sh

flux_dir=$gendata_dir/flux_decomposition
dhr=24
bl_scheme="MYNN25"
bl_scheme="YSU"

               
 
for day in 1 2; do

    for Ug in 20; do
        
        hrs_beg=$( printf "%02d" $(( $day * $dhr )) )
        hrs_end=$( printf "%02d" $(( ($day + 1) * $dhr )) )
       
        input_dir="${flux_dir}/hr${hrs_beg}-${hrs_end}"
        input_file="${input_dir}/case_mph-off_${bl_scheme}.nc"

        output_file=${fig_dir}/param_space_WRF_Ug${Ug}_hr${hrs_beg}-${hrs_end}_${bl_scheme}.svg

        eval "python3 src/plot_param_space_WRF_new.py  \
            --input $input_file  \
            --output ${output_file} \
            --varying-params Lx dSST \
            --param1-rng 0 200 \
            --param2-rng 0 1 \
            --fixed-params Ug \
            --fixed-param-values $Ug \
            --title 'Time: ${hrs_beg}-${hrs_end} hr ${bl_scheme}' \
            --no-display
        "
    done
done

echo "Done"
