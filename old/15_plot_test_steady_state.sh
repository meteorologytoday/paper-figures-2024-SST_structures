#!/bin/bash

source 00_setup.sh

nproc=2
output_fig_dir=$fig_dir/test_steady_state
dhr=120
offset=0

#dhr=6

#for dT in "000" "020" "040" "060" "080" "100"; do
for dT in "100"; do
    for Lx in "100" ; do
        for U in "20" ; do
            for _bl_scheme in "${bl_schemes[@]}" ; do
                
                casename=case_mph-off_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}
                input_dir=$data_dir/$target_lab/$casename
                output_dir=$output_fig_dir/$target_lab/Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}

                mkdir -p $output_dir

                #for t in $( seq 0 7 ); do
                for t in 0; do
                
                    hrs_beg=$( printf "%02d" $(( $offset + $t * $dhr )) )
                    hrs_end=$( printf "%02d" $(( $offset + ($t + 1) * $dhr )) )

                    output_name="$output_dir/steady_state_test_${hrs_beg}-${hrs_end}_${casename}.png"
                    extra_title=""

                    extra_title="$_bl_scheme. "
             
                    python3 src/test_steady_state.py \
                        --input-dir $input_dir  \
                        --exp-beg-time "2001-01-01 00:00:00" \
                        --wrfout-data-interval 60            \
                        --frames-per-wrfout-file 60          \
                        --time-rng $hrs_beg $hrs_end         \
                        --extra-title "$extra_title"         \
                        --coarse-grained-time $(( 3600 ))  \
                        --enclosed-time-rng 48 72            \
                        --diag-press-lev 850 \
                        --no-display \
                        --output $output_name

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

wait
