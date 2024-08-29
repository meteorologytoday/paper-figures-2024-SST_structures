#!/bin/bash

source 00_setup.sh



time_avg_interval=60   # minutes

hrs_beg=$(( 24 * 5 ))
hrs_end=$(( 24 * 10 ))


thumbnail_skip=2
#for bl_scheme in MYNN25 MYJ YSU; do
for bl_scheme in MYNN25 ; do
for target_lab in lab_FIXEDDOMAIN_SST_sine_WETLWSW ; do

    output_dir=$fig_dir/spectral_analysis_tracking_wnm${tracking_wnm}
    mkdir -p $output_dir
    
    input_dirs_base=""
    input_dirs=""
    labels=""
    wnms=""
    for Ug in 20 ; do
    for wnm in 004 005 007 010 020 040 ; do
    for dT in 100; do
    
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

        wnms="$wnms $wnm"

    done
    done
    done

    linestyles=(
        "solid"
        "dashed"
        "dotted"
        "dashdot"
        "solid"
        "dashed"
    )

    linecolors=(
        "black"
        "black"
        "black"
        "red"
        "blue"
        "blue"
    )

    output_file=$output_dir/spectral_analysis_${target_lab}_dT${dT}_${bl_scheme}_hr${hrs_beg}-${hrs_end}.svg

    eval "python3 src/plot_spectral_analysis_trace_wnm.py    \
        --input-dirs $input_dirs                   \
        --input-dirs-base $input_dirs_base         \
        --output $output_file                      \
        --no-display                               \
        --tracking-wnms $wnms                      \
        --time-rng $hrs_beg $hrs_end               \
        --exp-beg-time '2001-01-01 00:00:00'       \
        --time-rng $hrs_beg $hrs_end               \
        --wrfout-data-interval 3600                \
        --frames-per-wrfout-file 12                \
        --number-of-harmonics 22                   \
        --labeled-wvlen 100 200 500                \
        --linestyles ${linestyles[@]}              \
        --linecolors ${linecolors[@]}              \
        --thumbnail-skip $thumbnail_skip           \
        --varnames SST TOA QOA CH UA VA 
    "

    thumbnail_skip=$(( $thumbnail_skip + 2 ))

done
done