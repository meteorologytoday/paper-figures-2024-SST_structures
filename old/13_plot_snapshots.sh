#!/bin/bash

source 00_setup.sh

dhr=$(( 24 * 5 ))
output_fig_dir=$fig_dir/snapshots_dhr-${dhr}

nproc=20

proc_cnt=0



target_labs=(
    lab_sine_dry
    lab_sine_wet
    lab_sine_wetlw
    lab_sine_wetlwsw
)

bl_schemes=(
    MYNN25
)

trap "exit" INT TERM
trap "echo 'Exiting... ready to kill jobs... '; kill 0" EXIT


for dT in 300; do
    for Lx in 500 100 ; do
        for U in "20" ; do
            for target_lab in "${target_labs[@]}" ; do
                for _bl_scheme in "${bl_schemes[@]}" ; do
                    
                    if [[ "$target_lab" =~ "semiwet" ]]; then
                        mph=off
                        W_levs=( -10 10 11 )
                    elif [[ "$target_lab" =~ "wet" ]]; then
                        mph=on
                        W_levs=( -50 50 11 )
                    elif [[ "$target_lab" =~ "dry" ]]; then
                        mph=off
                        W_levs=( -2 2 21 )
                    fi




                    input_dir=$data_sim_dir/$target_lab/case_mph-${mph}_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}
                    output_dir=$output_fig_dir/$target_lab/Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}

                    mkdir -p $output_dir

                    #for t in $( seq 0 14 ); do
                    for t in 0 1 2; do
                    #for t in 2 ; do
                     
                        hrs_beg=$( printf "%02d" $(( $t * $dhr )) )
                        hrs_end=$( printf "%02d" $(( ($t + 1) * $dhr )) )

                        output_name="$output_dir/snapshot_${hrs_beg}-${hrs_end}.png"
                        extra_title=""

                        extra_title="$_bl_scheme. "
                 
                        python3 src/plot_snapshot_new.py  \
                            --input-dir $input_dir  \
                            --exp-beg-time "2001-01-01 00:00:00" \
                            --wrfout-data-interval 60            \
                            --frames-per-wrfout-file 60          \
                            --time-rng $(( $hrs_beg * 60 )) $(( $hrs_end * 60 ))  \
                            --extra-title "$extra_title"         \
                            --z-rng 0 5000 \
                            --U10-rng -5 5 \
                            --Q-rng -2 15 \
                            --W-levs "${W_levs[@]}" \
                            --SST-rng 11 19 \
                            --output $output_name \
                            --tke-analysis TRUE \
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
