#!/bin/bash

source 00_setup.sh
    

beg_days=(
    5
)

dhr=$(( 24 * 5 ))
output_fig_dir=$fig_dir/snapshots-full_dhr-${dhr}

nproc=5

proc_cnt=0

target_labs=(
    lab_FULL
    lab_SIMPLE
)

bl_schemes=(
    MYNN25
#    MYJ
#    YSU
)

source 98_trapkill.sh

for dT in 000; do
for wnm in 000 ; do
for U in "20" ; do
for target_lab in "${target_labs[@]}" ; do
for _bl_scheme in "${bl_schemes[@]}" ; do
 


    if [[ "$target_lab" =~ "FULL" ]]; then
        mph=on
        thumbnail_skip=6
    elif [[ "$target_lab" =~ "SIMPLE" ]]; then
        mph=off
        thumbnail_skip=0
    fi

    if [[ "$_bl_scheme" = "MYNN25" ]]; then
        tke_analysis=TRUE 
    elif [[ "$_bl_scheme" = "YSU" ]]; then
        tke_analysis=FALSE 
    elif [[ "$_bl_scheme" = "MYJ" ]]; then
        tke_analysis=FALSE 
    fi 

    if [[ "$target_lab" =~ "FULL" ]]; then
        exp_name="FULL"
    elif [[ "$target_lab" =~ "SIMPLE" ]]; then
        exp_name="SIMPLE"
    fi

    exp_name="${exp_name}."


    tke_analysis=TRUE

    input_dir=$gendata_dir/preavg/$target_lab/case_mph-${mph}_wnm${wnm}_U${U}_dT${dT}_${_bl_scheme}
    output_dir=$output_fig_dir/$target_lab/case_mph-${mph}_wnm${wnm}_U${U}_dT${dT}_${_bl_scheme}

    mkdir -p $output_dir

    for beg_day in "${beg_days[@]}"; do
     
        hrs_beg=$( printf "%02d" $(( $beg_day * 24 )) )
        hrs_end=$( printf "%02d" $(( $hrs_beg + $dhr )) )

        output="$output_dir/snapshot-full-vertical-profile_${hrs_beg}-${hrs_end}.svg"
        extra_title=""

        extra_title="${exp_name}${_bl_scheme}."

        THETA_rng=(285 310)
        Nfreq2_rng=( -0.2  8 )
        U_rng=(-5 25)
        TKE_rng=(-0.2 1.5)
        DTKE_rng=(-0.008 0.008)
        Q_rng=(-1 10)


        eval "python3 src/plot_system_state_delta.py  \
            --input-dir $input_dir  \
            --exp-beg-time '2001-01-01 00:00:00' \
            --wrfout-data-interval 3600          \
            --frames-per-wrfout-file 12          \
            --time-rng $(( $hrs_beg * 60 )) $(( $hrs_end * 60 ))  \
            --extra-title '$extra_title'         \
            --part2-z-rng 0 5000 \
            --part2-THETA-rng ${THETA_rng[@]} \
            --part2-Nfreq2-rng ${Nfreq2_rng[@]} \
            --part2-TKE-rng ${TKE_rng[@]} \
            --part2-DTKE-rng ${DTKE_rng[@]} \
            --part2-U-rng ${U_rng[@]} \
            --part2-Q-rng ${Q_rng[@]} \
            --output2 $output \
            --thumbnail-skip-part2 $thumbnail_skip \
            --part2-tke-analysis $tke_analysis \
            --plot-part2 \
            --no-display 
        " &








        proc_cnt=$(( $proc_cnt + 1))
        
        if (( $proc_cnt >= $nproc )) ; then
            echo "Max proc reached: $nproc"
            wait
            proc_cnt=0
        fi
    done
done
done
done
done
done

wait
