#!/bin/bash

source 00_setup.sh
source 98_trapkill.sh
nproc=10

time_avg_interval=60   # minutes

hrs_beg=$(( 24 * 5 ))
hrs_end=$(( 24 * 10 ))


thumbnail_skip=0
for wnm in 010 004 ; do
for bl_scheme in MYNN25 ; do
for target_lab in lab_FIXEDDOMAIN_SST_sine_WETLWSW ; do

    output_dir=$fig_dir/coherence_analysis
    
    mkdir -p $output_dir
    
    input_dirs_base=""
    input_dirs=""
    labels=""
    dSSTs=""
    for Ug in 20 ; do
    for dT in 000 010 030 050 100 150 200 250 300; do
    #for dT in 100 300; do
    #for dT in 000 300; do
    
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

        dSSTs="$dSSTs $( python3 -c "print(\"%.2f\" % ( float(\"$dT\") / 1e2 ) )" )"

        labels="$labels $dT"
    done
    done
    done
        
    varnames=(
        TOA QOA CH CQ UA VA
    )

    linestyles=(
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
    )


    output_file=$output_dir/coherence_on_dSST_vary_dSST_${target_lab}_wnm${wnm}_${bl_scheme}_hr${hrs_beg}-${hrs_end}.svg
    L=$( python3 -c "import numpy; print(\"%d\" % ( numpy.round( float('$Lx') / float('$wnm') ), ) )" )
    echo "L = $L"

    eval "python3 src/plot_coherence_vary_dSST.py  \
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
        --tracking-wnm $wnm                        \
        --dSSTs $dSSTs                             \
        --varnames "${varnames[@]}"                \
        --labels ${labels[@]}                      \
        --ctl-varname SST                          \
        --thumbnail-skip $thumbnail_skip           \
        --thumbnail-title \"$L km\"
    " &

    thumbnail_skip=$(( $thumbnail_skip + 1 ))

    nproc_cnt=$(( $nproc_cnt + 1 ))
    if (( $nproc_cnt >= $nproc )) ; then
        echo "Max batch_cnt reached: $nproc"
        wait
        batch_cnt=0
    fi

done
done

wait
echo "Done."
