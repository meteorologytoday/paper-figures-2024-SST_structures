#!/bin/bash

source 00_setup.sh


beg_days=(
    5
)

dhr=$(( 24 * 5 ))
output_fig_dir=$fig_dir/snapshots_dhr-${dhr}

nproc=1

proc_cnt=0

target_labs=(
    lab_FIXEDDOMAIN_SST_sine_DRY
    lab_FIXEDDOMAIN_SST_sine_WETLWSW
)

bl_schemes=(
    MYNN25
#    MYJ
#    YSU
)

source 98_trapkill.sh

for dT in 300; do
for wnm in 004 010 ; do #020 ; do
for U in "20" ; do
for target_lab in "${target_labs[@]}" ; do
for _bl_scheme in "${bl_schemes[@]}" ; do
 
    thumbnail_skip_part1=0
    thumbnail_skip_part2=0

    if [[ "$target_lab" =~ "SEMIWET" ]]; then
        mph=off
        #W_levs=( -10 10 11 )
        W_levs=( -2 2 21 )
        thumbnail_skip_part1=4
        thumbnail_skip_part2=6
    elif [[ "$target_lab" =~ "WET" ]]; then
        mph=on
        #W_levs=( -50 50 11 )
        W_levs=( -2 2 21 )
        thumbnail_skip_part1=4
        thumbnail_skip_part2=6
    elif [[ "$target_lab" =~ "DRY" ]]; then
        mph=off
        W_levs=( -2 2 21 )
    fi

    if [[ "$_bl_scheme" = "MYNN25" ]]; then
        tke_analysis=TRUE 
    elif [[ "$_bl_scheme" = "YSU" ]]; then
        tke_analysis=FALSE 
    elif [[ "$_bl_scheme" = "MYJ" ]]; then
        tke_analysis=FALSE 
    fi 

    if [[ "$target_lab" =~ "WETLWSW" ]]; then
        exp_name="FULL."
    elif [[ "$target_lab" =~ "DRY" ]]; then
        exp_name="SIMPLE."
    fi


    tke_analysis=FALSE

    input_dir=$gendata_dir/preavg/$target_lab/case_mph-${mph}_wnm${wnm}_U${U}_dT${dT}_${_bl_scheme}
    output_dir=$output_fig_dir/$target_lab/case_mph-${mph}_wnm${wnm}_U${U}_dT${dT}_${_bl_scheme}
    
    input_dir_base=$gendata_dir/preavg/$target_lab/case_mph-${mph}_wnm000_U${U}_dT000_${_bl_scheme}

    mkdir -p $output_dir

    x_rng=$( python3 -c "print(\"%.3f\" % ( $Lx / float('${wnm}'.lstrip('0')) , ))" )


    for beg_day in "${beg_days[@]}"; do
     
        hrs_beg=$( printf "%02d" $(( $beg_day * 24 )) )
        hrs_end=$( printf "%02d" $(( $hrs_beg + $dhr )) )

        output1_name="$output_dir/snapshot-part1_${hrs_beg}-${hrs_end}.svg"
        output2_name="$output_dir/snapshot-part2_${hrs_beg}-${hrs_end}.svg"
        extra_title=""

        extra_title="${exp_name}${_bl_scheme}."
 
        python3 src/plot_system_state_delta.py  \
            --input-dir $input_dir  \
            --input-dir-base $input_dir_base  \
            --exp-beg-time "2001-01-01 00:00:00" \
            --wrfout-data-interval 3600          \
            --frames-per-wrfout-file 12          \
            --time-rng $(( $hrs_beg * 60 )) $(( $hrs_end * 60 ))  \
            --extra-title "$extra_title"         \
            --z1-rng 0 5000 \
            --z2-rng 0 5000 \
            --U10-rng -5 5 \
            --Q-rng -3 3 \
            --W-levs "${W_levs[@]}" \
            --SST-rng -5 5 \
            --U-rng -1 1   \
            --V-rng -1 1   \
            --x-rng 0 $x_rng        \
            --x-rolling 11          \
            --output1 $output1_name \
            --output2 $output2_name \
            --thumbnail-skip-part1 $thumbnail_skip_part1 \
            --thumbnail-skip-part2 $thumbnail_skip_part2 \
            --tke-analysis $tke_analysis \
            --no-display &



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
