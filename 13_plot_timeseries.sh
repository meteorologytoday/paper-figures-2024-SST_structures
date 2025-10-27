#!/bin/bash

source 00_setup.sh
source 98_trapkill.sh

nproc_cnt_limit=1

output_fig_dir=$fig_dir/timeseries

for _dir in $output_fig_dir ; do
    echo "mkdir -p $_dir"
    mkdir -p $_dir
done


offset=$(( 24 * 0 ))
dhr=$(( 24 * 15 ))




for U in "${Us[@]}" ; do
for bl_scheme in MYNN25 MYJ YSU ; do
for smooth in 25 ; do
for wnm in 010 ; do
for dT in 100 ; do

    preavg_dir=$( gen_preavg_dir $U )
    input_dirs=(
        $preavg_dir/lab_SIMPLE/case_mph-off_wnm000_U${U}_dT000_${bl_scheme}
        $preavg_dir/lab_SIMPLE/case_mph-off_wnm${wnm}_U${U}_dT${dT}_${bl_scheme}
        $preavg_dir/lab_FULL/case_mph-on_wnm000_U${U}_dT000_${bl_scheme}
        $preavg_dir/lab_FULL/case_mph-on_wnm${wnm}_U${U}_dT${dT}_${bl_scheme}
    )


    linestyles=(
        "dashed"
        "solid"
        "dashed"
        "solid"
    )

    linecolors=(
        "black"
        "black"
        "red"
        "red"
    )

    labels=(
        "SIMPLE-dSST=0K"
        "SIMPLE-dSST=1K"
        "FULL-dSST=0K"
        "FULL-dSST=1K"
    )

    hrs_beg=$( printf "%03d" $(( $offset )) )
    hrs_end=$( printf "%03d" $(( $offset + $dhr )) )



    echo "Doing diagnostic simple"
    output="$output_fig_dir/timeseries_wnm${wnm}_U${U}_dT${dT}_${bl_scheme}_timeseries_smooth-${smooth}_${hrs_beg}-${hrs_end}.svg"
    extra_title=""
    python3 src/plot_timeseries.py \
        --input-dirs "${input_dirs[@]}"      \
        --linestyles "${linestyles[@]}"      \
        --linecolors "${linecolors[@]}"      \
        --labels "${labels[@]}"              \
        --exp-beg-time "2001-01-01 00:00:00" \
        --wrfout-data-interval 3600          \
        --frames-per-wrfout-file 12          \
        --time-rng $hrs_beg $hrs_end         \
        --extra-title "$extra_title"         \
        --tick-interval-hour 24              \
        --ncols 3                            \
        --smooth $smooth                     \
        --no-display                         \
        --varnames    PBLH    TA     QA      \
                      PRECIP  HFX    LH      \
                      WND_sfc CH     CD0     \
        --output $output & 

        nproc_cnt=$(( $nproc_cnt + 1))
        
        if (( $nproc_cnt >= $nproc_cnt_limit )) ; then
            echo "Max nproc_cnt reached: $nproc_cnt"
            wait
            nproc_cnt=0
        fi
     
done
done
done
done
done

wait

echo "PLOTTING DONE."
