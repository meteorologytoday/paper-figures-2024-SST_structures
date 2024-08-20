#!/bin/bash

source 00_setup.sh

output_dir=$fig_dir/transfer_function

time_avg_interval=60   # minutes


hrs_beg=$(( 24 * 5 ))
hrs_end=$(( 24 * 10 ))

mkdir -p $output_dir
    
#for bl_scheme in MYNN25 MYJ YSU; do
for bl_scheme in MYNN25 ; do
for target_lab in lab_FIXEDDOMAIN_SST_sine_WETLWSW ; do
    
    input_dirs_base=""
    input_dirs=""
    labels=""

    for Ug in 20 ; do
    for wnm in 004 ; do
    for dT in 100 200 300; do
    #for dT in 300; do
    
        if [[ "$target_lab" =~ "SEMIWET" ]]; then
            mph=off
        elif [[ "$target_lab" =~ "WET" ]]; then
            mph=on
        elif [[ "$target_lab" =~ "DRY" ]]; then
            mph=off
        fi

        casename="case_mph-${mph}_wnm${wnm}_U${Ug}_dT${dT}_${bl_scheme}"
        casename_base="case_mph-${mph}_wnm000_U${Ug}_dT000_${bl_scheme}"

        input_dir_root=$gendata_dir/preavg/$target_lab
        
        input_dir="$input_dir_root/$casename"
        input_dirs="$input_dirs $input_dir"

        input_dir_base="$input_dir_root/$casename_base"
        input_dirs_base="$input_dirs_base $input_dir_base"

        labels="$labels nu${wnm}-dSST${dT}"

    done
    done
    done

    output_file=$output_dir/transfer_function_${target_lab}_${bl_scheme}_wnm${wnm}_hr${hrs_beg}-${hrs_end}.svg

    eval "python3 src/plot_transfer_function.py    \
        --input-dirs $input_dirs                   \
        --input-dirs-base $input_dirs_base         \
        --output $output_file                      \
        --no-display                               \
        --time-rng $hrs_beg $hrs_end               \
        --exp-beg-time '2001-01-01 00:00:00'       \
        --time-rng $hrs_beg $hrs_end               \
        --wrfout-data-interval 3600                \
        --frames-per-wrfout-file 12                \
        --number-of-harmonics 22                   \
        --labeled-wvlen 100 200 500             \
        --labels $labels \
        --varnames SST TA UA 
    "



done
done
