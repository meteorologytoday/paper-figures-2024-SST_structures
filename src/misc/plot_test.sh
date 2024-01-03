#!/bin/bash

fig_dir=figures_snapshot_test

nproc=5

proc_cnt=0

for dT in "100"; do
    for _ML in "woML" ; do
        for _bl_scheme in "MYNN25" "YSU" ; do
            input_dir=../11_SST_gaussian/lab_gaussian/case_mph-off_dT${dT}_wid050_${_ML}_${_bl_scheme}_wpkt01

            output_dir=$fig_dir/dT${dT}/${_ML}_${_bl_scheme}

            mkdir -p $output_dir

            #for t in $( seq 0 7 ); do
            for t in 5; do
            
                hrs_beg=$( printf "%02d" $(( $t * 6 )) )
                hrs_end=$( printf "%02d" $(( ($t + 1) * 6 )) )

                output_name="$output_dir/comparison_${_ML}_${hrs_beg}-${hrs_end}.png"
                extra_title=""

                if [ "$_ML" = "wML" ] ; then
                    extra_title="$extra_title with ML"
                elif [ "$_ML" = "woML" ] ; then
                    extra_title="$extra_title without ML"
                fi

                extra_title="$extra_title, $_bl_scheme. "
         
                python3 plot_snapshot.py  \
                    --input-dir $input_dir  \
                    --exp-beg-time "2001-01-01 00:00:00" \
                    --wrfout-data-interval 60            \
                    --frames-per-wrfout-file 60          \
                    --time-rng $(( $hrs_beg * 60 )) $(( $hrs_end * 60 ))  \
                    --extra-title "$extra_title"         \
                    --x-rng 0 1000         \
                    --z-rng 0 1500         \
                    --U10-rng -.5 2.5       \
                    --output $output_name \
                    --no-display &

                proc_cnt=$(( $proc_cnt + 1))
                
                if (( $proc_cnt >= $nproc )) ; then
                    echo "Max proc reached: $nproc"
                    wait
                    proc_cnt=0
                fi
            done

            if [ ] ; then
            hrs_beg=0
            hrs_end=72
            output_name="$output_dir/hovmoeller_${_ML}_${hrs_beg}-${hrs_end}.png"
            python3 plot_hovmoeller_new.py  \
                --input-dir $input_dir  \
                --exp-beg-time "2001-01-01 00:00:00" \
                --wrfout-data-interval 60            \
                --frames-per-wrfout-file 60          \
                --time-rng $(( $hrs_beg * 60 )) $(( $hrs_end * 60 ))  \
                --extra-title "$extra_title"         \
                --x-rng 0 1000         \
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


wait
