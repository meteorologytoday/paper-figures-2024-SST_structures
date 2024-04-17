#!/bin/bash

source 00_setup.sh

nproc=2
output_fig_dir=$fig_dir/timeseries
dhr=120
offset=0

#dhr=6

#for dT in "000" "020" "040" "060" "080" "100"; do
for dT in "100" "020"; do
    for Lx in "100" "200" ; do
        for U in "20" "10" ; do
            for _bl_scheme in "MYNN25" ; do
                
                input_dir=$data_dir/$target_lab/case_mph-off_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}
                output_dir=$output_fig_dir/Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}

                mkdir -p $output_dir

                #for t in $( seq 0 7 ); do
                for t in 0; do
                
                    hrs_beg=$( printf "%02d" $(( $offset + $t * $dhr )) )
                    hrs_end=$( printf "%02d" $(( $offset + ($t + 1) * $dhr )) )

                    output_name="$output_dir/TOAQOA_timeseries_${hrs_beg}-${hrs_end}.png"
                    extra_title=""

                    extra_title="$_bl_scheme. "
             
                    python3 src/timeseries.py \
                        --input-dir $input_dir  \
                        --exp-beg-time "2001-01-01 00:00:00" \
                        --wrfout-data-interval 60            \
                        --frames-per-wrfout-file 60          \
                        --time-rng $hrs_beg $hrs_end         \
                        --extra-title "$extra_title"         \
                        --coarse-grained-time 3600           \
                        --no-display \
                        --output $output_name &

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
