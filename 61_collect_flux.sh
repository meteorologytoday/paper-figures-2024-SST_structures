#!/bin/bash

source 00_setup.sh

flux_dir=$gendata_dir/flux_decomposition

dhr=24

Lxs="50 100 200 300 400 500"
Ugs="20"
dSSTs="000 050 100 150 200 250 300"


hrs_beg=$(( 24 * 10 ))
hrs_end=$(( 24 * 15 ))

analysis_root=$gendata_dir/analysis_timeseries


fixed_Lx=500
fixed_dSST=300


for fixed_param in Lx dSST ; do

    echo "Fixed Param: ${fixed_param}"
    output_root=$gendata_dir/collected_flux/fixed_${fixed_param}

    if [[ "$fixed_param" =~ "Lx" ]] ; then
        
        _Lxs=$fixed_Lx
        _dSSTs=$dSSTs

    elif [[ "$fixed_param" =~ "dSST" ]] ; then
        
        _Lxs=$Lxs
        _dSSTs=$fixed_dSST

    else 
    
        echo "ERROR: unknown fixed_param : $fixed_param"

    fi




    for _bl_scheme in YSU MYJ MYNN25 ; do
    #for _bl_scheme in MYNN25 YSU MYJ; do
    for target_lab in lab_sine_WETLWSW ; do       
        
        if [[ "$target_lab" =~ "SEMIWET" ]]; then
            mph=off
        elif [[ "$target_lab" =~ "WET" ]]; then
            mph=on
        elif [[ "$target_lab" =~ "DRY" ]]; then
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
           
            set -x 
            eval "python3 src/collect_flux_analysis.py  \
                --input-dir-fmt $input_dir_fmt  \
                --output $output_file \
                --Ugs $Ugs \
                --Lxs $_Lxs \
                --dSSTs $_dSSTs \
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

done   
 
echo "Done"
