#!/bin/bash

source 00_setup.sh

flux_dir=$gendata_dir/flux_decomposition

dhr=24

Lxs="20 40 60 80 100 120 140 160 180 200"
Ugs="5 10 15 20"
dSSTs="100"
bl_scheme="MYNN25"

                
for day in 1; do
    
    hrs_beg=$( printf "%02d" $(( $day * $dhr )) )
    hrs_end=$( printf "%02d" $(( ($day + 1) * $dhr )) )
   
    input_dir="${flux_dir}/hr${hrs_beg}-${hrs_end}"
    output_file="${input_dir}/case_mph-off_${bl_scheme}.nc"

    if [ ! -f "$output_file" ] ; then

        eval "python3 src/collect_flux_analysis.py  \
            --input-dir $input_dir  \
            --output ${output_file} \
            --file-fmt 'case_mph-off_Lx{Lx:03d}_U{Ug:02d}_dT{dSST:03d}_MYNN25.nc' \
            --Ugs $Ugs \
            --Lxs $Lxs \
            --dSSTs $dSSTs
        "
    fi
    
done

echo "Done"
