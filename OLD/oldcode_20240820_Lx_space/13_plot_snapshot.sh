#!/bin/bash

source 00_setup.sh

dhr=$(( 24 * 5 ))
output_fig_dir=$fig_dir/snapshots_dhr-${dhr}

nproc=1

proc_cnt=0

target_labs=(
    lab_FIXEDDOMAIN_SST_sine_WETLWSW
    lab_FIXEDDOMAIN_SST_sine_DRY
)

bl_schemes=(
    MYNN25
    MYJ
    YSU
)

source 98_trapkill.sh

for dT in 300; do
#    for Lx in 500 100 ; do
    for wnm in 004 020 ; do
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

                    #input_dir=$data_sim_dir/$target_lab/case_mph-${mph}_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}
                    input_dir=$gendata_dir/preavg/$target_lab/case_mph-${mph}_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}
                    output_dir=$output_fig_dir/$target_lab/Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}
                    
                    input_dir_base=$gendata_dir/preavg/$target_lab/case_mph-${mph}_Lx${Lx}_U${U}_dT000_${_bl_scheme}

                    mkdir -p $output_dir

                    for t in 1 ; do #2 1 0; do
                     
                        hrs_beg=$( printf "%02d" $(( $t * $dhr )) )
                        hrs_end=$( printf "%02d" $(( ($t + 1) * $dhr )) )

                        output1_name="$output_dir/snapshot-part1_${hrs_beg}-${hrs_end}.svg"
                        output2_name="$output_dir/snapshot-part2_${hrs_beg}-${hrs_end}.svg"
                        extra_title=""

                        extra_title="${exp_name}${_bl_scheme}."
                 
                        python3 src/plot_snapshot_split_frame.py  \
                            --input-dir $input_dir  \
                            --input-dir-base $input_dir_base  \
                            --exp-beg-time "2001-01-01 00:00:00" \
                            --wrfout-data-interval 3600          \
                            --frames-per-wrfout-file 12          \
                            --time-rng $(( $hrs_beg * 60 )) $(( $hrs_end * 60 ))  \
                            --extra-title "$extra_title"         \
                            --z1-rng 0 5000 \
                            --z2-rng 0 2000 \
                            --U10-rng -5 5 \
                            --Q-rng -2 15 \
                            --W-levs "${W_levs[@]}" \
                            --SST-rng 11 19 \
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
