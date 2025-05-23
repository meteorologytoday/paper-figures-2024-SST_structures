#!/bin/bash

source 00_setup.sh
source 98_trapkill.sh
nproc=1

time_avg_interval=60   # minutes

hrs_beg=$(( 24 * 5 ))


thumbnail_skip=0
for bl_scheme in MYNN25 MYJ YSU ; do

    dhr=$( get_dhr $bl_scheme ) 
    hrs_end=$(( $hrs_beg + $dhr ))

    input_files=""
    labels=""
    for dT in 100 ; do
    for target_lab_suffix in SIMPLE FULL ; do

        target_lab=lab_${target_lab_suffix}
    
        input_file=$gendata_dir/coherence_analysis/coherence_analysis_dSST_vary_wnm_${target_lab}_dSST${dT}_${bl_scheme}_hr${hrs_beg}-${hrs_end}.nc

        input_files="$input_files $input_file"
        labels="$labels ${bl_scheme}-${target_lab_suffix}"
    done
    done


    varnames=(
        UA
    )

    linestyles=(
        "solid"
        "dashed"
    )

    linecolors=(
        "black"
        "black"
    )


    output_dir=$fig_dir/coherence_analysis
    
    mkdir -p $output_dir
    
    output_file=$output_dir/coherence_on_dSST_vary_wnm_dSST${dT}_${bl_scheme}_hr${hrs_beg}-${hrs_end}.svg

    python3 src/plot_coherence.py \
        --input-files ${input_files[@]} \
        --output $output_file                      \
        --no-display                               \
        --labels $labels \
        --linestyles ${linestyles[@]}              \
        --linecolors ${linecolors[@]}              \
        --varnames "${varnames[@]}"                \
        --ctl-varname SST                          \
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

