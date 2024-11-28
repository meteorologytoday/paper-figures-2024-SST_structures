#!/bin/bash

source 00_setup.sh
source 98_trapkill.sh
nproc=1

time_avg_interval=60   # minutes

hrs_beg=$(( 24 * 5 ))
hrs_end=$(( 24 * 10 ))


thumbnail_skip=0
for dT in 100 ; do
for bl_scheme in MYNN25 ; do
for target_lab in lab_FIXEDDOMAIN_SST_sine_DRY lab_FIXEDDOMAIN_SST_sine_WETLWSW; do

    output_dir=$fig_dir/linearity_analysis
    
    mkdir -p $output_dir
    
    input_dirs_base=""
    input_dirs=""
    labels=""
    dSSTs=""
    tracking_wnms=""
    for Ug in 20 ; do
    for wnm in 004 005 007 010 020 040; do
    #for wnm in 004 005 ; do
    
        if [[ "$target_lab" =~ "SEMIWET" ]]; then
            mph=off
        elif [[ "$target_lab" =~ "WET" ]]; then
            mph=on
            title="FULL"
        elif [[ "$target_lab" =~ "DRY" ]]; then
            mph=off
            title="SIMPLE"
        fi

        casename="case_mph-${mph}_wnm${wnm}_U${Ug}_dT${dT}_${bl_scheme}"
        casename_base="case_mph-${mph}_wnm000_U${Ug}_dT000_${bl_scheme}"

        input_dir_root=$gendata_dir/preavg/$target_lab
        
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

    output_file=$output_dir/linearity_vary_wnm_${target_lab}_dSST${dT}_${bl_scheme}_hr${hrs_beg}-${hrs_end}.svg

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
        --ylim 0.7 1.02                            \
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
