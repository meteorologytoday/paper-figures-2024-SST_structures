#!/bin/bash

source 00_setup.sh

output_fig_dir=$fig_dir/snapshots_15min_new

nproc=20

proc_cnt=0

dmin=15

for dT in "100"; do
    for Lx in "100" ; do
        for U in "15" ; do
            for _bl_scheme in "MYNN25" ; do
                
                input_dir=$data_dir/$target_lab/case_mph-off_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}
                output_dir=$output_fig_dir/Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}

                mkdir -p $output_dir

                #for t in $( seq 0 7 ); do
                #for t in $( seq 0 480 ) ; do
                for t in $( seq 240 280 ) ; do
                 
                    min_beg=$( printf "%04d" $(( $t * $dmin )) )
                    min_end=$( printf "%04d" $(( ($t + 1) * $dmin )) )

                    output_name="$output_dir/snapshot_${min_beg}-${min_end}.png"
                    extra_title=""

                    extra_title="$_bl_scheme. "
            
                    if [ -f "$output_name" ] ; then
                        echo "Already exists. Skip $output_name"
                    else

 
                        python3 src/plot_snapshot_new.py  \
                            --input-dir $input_dir  \
                            --exp-beg-time "2001-01-01 00:00:00" \
                            --wrfout-data-interval 60            \
                            --frames-per-wrfout-file 60          \
                            --time-rng $min_beg $min_end         \
                            --extra-title "$extra_title"         \
                            --z-rng 0 5000 \
                            --output $output_name \
                            --no-display &

                        proc_cnt=$(( $proc_cnt + 1))
                        
                        if (( $proc_cnt >= $nproc )) ; then
                            echo "Max proc reached: $nproc"
                            wait
                            proc_cnt=0
                        fi

                    fi
                done
            done
        done
    done
done
wait
