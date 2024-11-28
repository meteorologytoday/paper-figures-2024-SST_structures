#!/bin/bash

source 00_setup.sh

dhr=24

wnms="004 005 007 010 020 040"
Ugs="20"
dSSTs="000 050 100 150 200 250 300"


hrs_beg=$(( 24 * 5 ))
hrs_end=$(( 24 * 10 ))

analysis_root=$gendata_dir/dF_analysis_timeseries


fixed_wnm=004
fixed_dSST=300


#for fixed_param in dSST wnm ; do
#for fixed_param in dSST ; do
for fixed_param in wnm ; do


    echo "Fixed Param: ${fixed_param}"
    output_root=$gendata_dir/dF_collected_flux/fixed_${fixed_param}

    if [[ "$fixed_param" =~ "wnm" ]] ; then
        
        _wnms=$fixed_wnm
        _dSSTs=$dSSTs

    elif [[ "$fixed_param" =~ "dSST" ]] ; then
        
        _wnms=$wnms
        _dSSTs=$fixed_dSST

    else 
    
        echo "ERROR: unknown fixed_param : $fixed_param"

    fi




    #for _bl_scheme in YSU MYJ MYNN25 ; do
    for _bl_scheme in MYNN25 ; do #MYJ YSU; do
    for target_lab in lab_FIXEDDOMAIN_SST_sine_WETLWSW ; do       
        
        if [[ "$target_lab" =~ "SEMIWET" ]]; then
            mph=off
        elif [[ "$target_lab" =~ "WET" ]]; then
            mph=on
        elif [[ "$target_lab" =~ "DRY" ]]; then
            mph=off
        fi

        casename=case_mph-${mph}_wnm${wnm}_U${U}_dT${dT}_${_bl_scheme}
        input_dir_fmt=$analysis_root/$target_lab/case_mph-${mph}_wnm{wnm:s}_U{Ug:s}_dT{dSST:s}_${_bl_scheme}/avg_before_analysis-TRUE

        output_dir=$output_root/${target_lab}
        output_file=$output_dir/collected_flux_${_bl_scheme}_hr${hrs_beg}-${hrs_end}.nc

        mkdir -p $output_dir

        if [ -f "$output_file" ] ; then

            echo "File $output_file exists! Skip."

        else
           
            set -x 
            eval "python3 src/collect_flux_analysis_wnm.py  \
                --input-dir-fmt $input_dir_fmt  \
                --output $output_file \
                --Ugs $Ugs \
                --wnms $_wnms \
                --dSSTs $_dSSTs \
                --time-rng $hrs_beg $hrs_end \
                --moving-avg-cnt 25 \
                --expand-time-min $(( 24 * 60 )) \
                --exp-beg-time 2001-01-01T00:00:00 \
                --wrfout-data-interval 3600 \
                --frames-per-wrfout-file 12 \
                --no-extra-variable
            "
        fi

    done
    done

done   
 
echo "Done"
