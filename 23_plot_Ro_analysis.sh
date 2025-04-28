#!/bin/bash

source 00_setup.sh
source 98_trapkill.sh
nproc=1

time_avg_interval=60   # minutes

hrs_beg=$(( 24 * 5 ))
hrs_end=$(( 24 * 10 ))


thumbnail_skip=0
for bl_scheme in MYNN25 MYJ YSU ; do

    input_files=""
    labels=""
    for dT in 100 ; do
    for target_lab_suffix in SIMPLE FULL ; do

        target_lab=lab_${target_lab_suffix}
    
        input_file=$gendata_dir/Ro_analysis/Ro_analysis_vary_wnm_${target_lab}_dSST${dT}_${bl_scheme}_hr${hrs_beg}-${hrs_end}.nc

        input_files="$input_files $input_file"
        labels="$labels ${bl_scheme}-${target_lab_suffix}"
    done
    done


    linestyles=(
        "solid"
        "dashed"
    )

    linecolors=(
        "black"
        "black"
    )


    output_dir=$fig_dir/Ro_analysis
    
    mkdir -p $output_dir
    
    output_file=$output_dir/Ro_analysis_vary_wnm_dSST${dT}_${bl_scheme}_hr${hrs_beg}-${hrs_end}.svg

    python3 src/plot_Ro_analysis.py \
        --input-files ${input_files[@]} \
        --output $output_file                      \
        --no-display                               \
        --labels $labels \
        --linestyles ${linestyles[@]}              \
        --linecolors ${linecolors[@]}              \
        --thumbnail-skip $thumbnail_skip          & 

    thumbnail_skip=$(( $thumbnail_skip + 1 ))

    nproc_cnt=$(( $nproc_cnt + 1 ))
    if (( $nproc_cnt >= $nproc )) ; then
        echo "Max batch_cnt reached: $nproc"
        wait
        nproc_cnt=0
    fi

done

wait
echo "Done."

