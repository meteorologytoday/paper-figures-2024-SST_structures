#!/bin/bash

source 00_setup.sh

output_fig_dir=$fig_dir/snapshots_15min_new

nproc=50

proc_cnt=0

dmin=$(( 24 * 60 ))

trap "exit" INT TERM
trap "echo 'Exiting... ready to kill jobs... '; kill 0" EXIT

for target_lab in lab_sine_semiwet lab_sine_wet lab_sine_dry ; do
for dT in 100; do
    for Lx in 500 100 ; do
        for U in 20; do
            for _bl_scheme in "MYNN25" ; do
                
                
                if [[ "$target_lab" =~ "semiwet" ]]; then
                    mph=off
                elif [[ "$target_lab" =~ "wet" ]]; then
                    mph=on
                elif [[ "$target_lab" =~ "dry" ]]; then
                    mph=off
                fi



                input_dir=$data_sim_dir/$target_lab/case_mph-${mph}_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}
                output_dir=$output_fig_dir/$target_lab/Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}

                mkdir -p $output_dir

                for t in $( seq $(( ( 0 * 24 ) * 60 / $dmin )) $(( ( ( 20 * 24 ) * 60 / $dmin ) - 1 )) ) ; do
                 
                    min_beg=$( printf "%05d" $(( $t * $dmin )) )
                    min_end=$( printf "%05d" $(( ($t + 1) * $dmin )) )

                    output_name="$output_dir/snapshot_${min_beg}-${min_end}.png"
                    extra_title=""

                    extra_title="$_bl_scheme. "
            
                    if [ -f "$output_name" ] ; then
                        echo "Already exists. Skip $output_name"
                    else

                        if [ "$_bl_scheme" = "YSU" ]; then
                            tke_analysis=FALSE
                        else
                            tke_analysis=TRUE
                        fi 
                        python3 src/plot_snapshot_new.py  \
                            --input-dir $input_dir  \
                            --exp-beg-time "2001-01-01 00:00:00" \
                            --wrfout-data-interval 60            \
                            --frames-per-wrfout-file 60          \
                            --time-rng $min_beg $min_end         \
                            --extra-title "$extra_title"         \
                            --z-rng 0 2000 \
                            --SST-rng  11.5 18.5         \
                            --U-rng 10 30  \
                            --V-rng -1 4   \
                            --DTKE-rng   -0.01 0.01    \
                            --TKE-rng  0.0 2.0         \
                            --tke-analysis $tke_analysis           \
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
done
wait
