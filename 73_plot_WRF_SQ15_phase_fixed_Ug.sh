#!/bin/bash

source 00_setup.sh

flux_dir_WRF=$gendata_dir/flux_decomposition
dhr=24
bl_scheme="MYNN25"

               
 
for day in 2; do


    for Ug in 20 ; do
    for RH in 0.999 ; do
    for DTheta in 0.1 1 2 3 4 5 10 ; do
        
        hrs_beg=$( printf "%02d" $(( $day * $dhr )) )
        hrs_end=$( printf "%02d" $(( ($day + 1) * $dhr )) )
       
        input_dir_WRF="${flux_dir_WRF}/hr${hrs_beg}-${hrs_end}"
        input_file_WRF="${input_dir_WRF}/case_mph-off_${bl_scheme}.nc"
        
        input_file_SQ15="${gendata_dir}/phase_SQ15.nc"

        output_file=${fig_dir}/param_space_WRF_SQ15_Ug-${Ug}_DTheta-${DTheta}_RH-${RH}_hr${hrs_beg}-${hrs_end}.svg

        eval "python3 src/plot_param_space_WRF_SQ15_new.py  \
            --input-file-SQ15 $input_file_SQ15  \
            --input-file-WRF  $input_file_WRF   \
            --output ${output_file} \
            --fixed-RH $RH \
            --fixed-DTheta $DTheta \
            --fixed-Ug $Ug \
            --param1-rng 0 200 \
            --param2-rng 0 1 \
            --title 'Time: ${hrs_beg}-${hrs_end} hr' \
            --no-display
        "
    done
    done
    done
done

echo "Done"
