#!/bin/bash

source 00_setup.sh

target_lab=lab_sine_noadv


output_fig_dir=$fig_dir/snapshots_notkeadv

nproc=40

proc_cnt=0

dmin=6

trap "exit" INT TERM
trap "echo 'Exiting... ready to kill jobs... '; kill 0" EXIT

for dT in "000" "100"; do
    for Lx in "100" ; do
        for U in "20" ; do
            for _bl_scheme in "MYNN25" ; do
                
                input_dir=$data_dir/$target_lab/case_mph-off_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}
                output_dir=$output_fig_dir/Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}

                mkdir -p $output_dir

                #for t in $( seq 0 7 ); do
                for t in $( seq 0 $(( 120 * 60 / $dmin )) ) ; do
                #for t in $( seq 0 7 ) ; do
                #for t in $( seq 240 280 ) ; do
                 
                    min_beg=$( printf "%05d" $(( $t * $dmin )) )
                    min_end=$( printf "%05d" $(( ($t + 1) * $dmin )) )

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
                            --U-rng 10 30  \
                            --V-rng -1 4   \
                            --DTKE-rng   -0.01 0.01    \
                            --TKE-rng  0.0 2.0         \
                            --output $output_name    \
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
