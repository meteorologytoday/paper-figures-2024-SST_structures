#!/bin/bash

source 00_setup.sh

output_fig_dir=$fig_dir/snapshots

nproc=2

proc_cnt=0

dhr=24

for dT in "100"; do
    for Lx in "100" ; do
        for U in "15" ; do
            for _bl_scheme in "MYNN25" ; do
                
                input_dir=$data_dir/$target_lab/case_mph-off_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}
                output_dir=$output_fig_dir/Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}

                mkdir -p $output_dir

                #for t in $( seq 0 7 ); do
                for t in 1 2 0 ; do
                 
                    hrs_beg=$( printf "%02d" $(( $t * $dhr )) )
                    hrs_end=$( printf "%02d" $(( ($t + 1) * $dhr )) )

                    output_name="$output_dir/snapshot_${hrs_beg}-${hrs_end}.png"
                    extra_title=""

                    extra_title="$_bl_scheme. "
             
                    python3 src/plot_snapshot.py  \
                        --input-dir $input_dir  \
                        --exp-beg-time "2001-01-01 00:00:00" \
                        --wrfout-data-interval 60            \
                        --frames-per-wrfout-file 60          \
                        --time-rng $(( $hrs_beg * 60 )) $(( $hrs_end * 60 ))  \
                        --extra-title "$extra_title"         \
                        --z-rng 0 5000 \
                        --output $output_name \
                        --no-display

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
