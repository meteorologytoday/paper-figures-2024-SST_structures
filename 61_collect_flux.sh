#!/bin/bash

source 00_setup.sh

flux_dir=$gendata_dir/flux_decomposition

dhr=24

Lxs="20 40 60 80 100 120 140 160 180 200"
Ugs="0 5 10 15 20 25 30"
dSST="000 020 040 060 080 100 150 200 250 300"

Lxs="50 100 200 300 400 500"
Ugs="20"
dSSTs="000 050 100 150 200 250 300"


hrs_beg=$(( 24 * 10 ))
hrs_end=$(( 24 * 15 ))

analysis_root=$gendata_dir/analysis_timeseries
output_root=$gendata_dir/collected_flux

for _bl_scheme in MYNN25 ; do
for target_lab in lab_sine_wetlwsw ; do       
    
    if [[ "$target_lab" =~ "semiwet" ]]; then
        mph=off
    elif [[ "$target_lab" =~ "wet" ]]; then
        mph=on
    elif [[ "$target_lab" =~ "dry" ]]; then
        mph=off
    fi

    casename=case_mph-${mph}_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}
    input_dir_fmt=$analysis_root/$target_lab/case_mph-${mph}_Lx{Lx:s}_U{Ug:s}_dT{dSST:s}_${_bl_scheme}/avg_before_analysis-TRUE

    output_dir=$output_root/${target_lab}
    output_file=$output_dir/collected_flux_${_bl_scheme}_hr${hrs_beg}-${hrs_end}.nc

    mkdir -p $output_dir

    if [ -f "$output_file" ] ; then

        echo "File $output_file exists! Skip."

    else
        
        eval "python3 src/collect_flux_analysis.py  \
            --input-dir-fmt $input_dir_fmt  \
            --output $output_file \
            --Ugs $Ugs \
            --Lxs $Lxs \
            --dSSTs $dSSTs \
            --time-rng $hrs_beg $hrs_end \
            --moving-avg-cnt 25 \
            --expand-time-min $(( 24 * 60 )) \
            --exp-beg-time 2001-01-01T00:00:00 \
            --wrfout-data-interval 3600 \
            --frames-per-wrfout-file 12
        "
    fi

done
done
    
echo "Done"
