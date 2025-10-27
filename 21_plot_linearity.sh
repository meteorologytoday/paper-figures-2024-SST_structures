#!/bin/bash

source 00_setup.sh
source 98_trapkill.sh
nproc=1

time_avg_interval=60   # minutes

hrs_beg=$(( 24 * 10 ))



thumbnail_skip=0
for dT in 100 ; do
for bl_scheme in MYNN25 MYJ YSU ; do
for target_lab in lab_FULL lab_SIMPLE; do

    
    dhr=$( get_dhr $bl_scheme ) 
    hrs_end=$(( $hrs_beg + $dhr ))

    output_dir=$fig_dir/linearity_analysis
    
    mkdir -p $output_dir
    
    input_dirs_base=""
    input_dirs=""
    labels=""
    dSSTs=""
    tracking_wnms=""
    #for U in "${Us[@]}" ; do
    for U in 20 ; do
    for wnm in 004 005 007 010 020 040; do
    #for wnm in 004 005 ; do
    
        if [[ "$target_lab" =~ "FULL" ]]; then
            mph=on
            title="FULL"
        elif [[ "$target_lab" =~ "SIMPLE" ]]; then
            mph=off
            title="SIMPLE"
        fi

        casename="case_mph-${mph}_wnm${wnm}_U${U}_dT${dT}_${bl_scheme}"
        casename_base="case_mph-${mph}_wnm000_U${U}_dT000_${bl_scheme}"

        input_dir_root=$( gen_preavg_dir $U )/$target_lab
        
        input_dir="$input_dir_root/$casename"
        input_dirs="$input_dirs $input_dir"

        input_dir_base="$input_dir_root/$casename_base"
        input_dirs_base="$input_dirs_base $input_dir_base"

        tracking_wnms="$tracking_wnms $wnm"

    done
    done
        
    varnames=(
        TOA QOA CH CQ UA VA DIVA VORA
    )

    linestyles=(
        "solid"
        "dashed"
        "solid"
        "dashed"
        "solid"
        "dashed"
        "solid"
        "dashed"
    )

    linecolors=(
        "black"
        "black"
        "reddishpurple"
        "reddishpurple"
        "skyblue"
        "skyblue"
        "orange"
        "orange"
    )

    output_file=$output_dir/linearity_vary_wnm_${target_lab}_dSST${dT}_U${U}_${bl_scheme}_hr${hrs_beg}-${hrs_end}.svg

    eval "python3 src/plot_linearity_vary_wnm.py    \
        --input-dirs $input_dirs                   \
        --input-dirs-base $input_dirs_base         \
        --output $output_file                      \
        --no-display                               \
        --time-rng $hrs_beg $hrs_end               \
        --exp-beg-time '2001-01-01 00:00:00'       \
        --time-rng $hrs_beg $hrs_end               \
        --wrfout-data-interval 3600                \
        --frames-per-wrfout-file 12                \
        --labeled-wvlen 100 200 500                \
        --linestyles ${linestyles[@]}              \
        --linecolors ${linecolors[@]}              \
        --tracking-wnms ${tracking_wnms[@]}        \
        --varnames "${varnames[@]}"                \
        --thumbnail-skip $thumbnail_skip           \
        --ylim 0.9 1.02                            \
        --thumbnail-titles "$title"
    " &

    thumbnail_skip=$(( $thumbnail_skip + 1 ))

    nproc_cnt=$(( $nproc_cnt + 1 ))
    if (( $nproc_cnt >= $nproc )) ; then
        echo "Max batch_cnt reached: $nproc"
        wait
        nproc_cnt=0
    fi

done
done
done

wait
echo "Done."
