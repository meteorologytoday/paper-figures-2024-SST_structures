#!/bin/bash

dTs=(
    050
    100
)

fig_dir=figures




nproc=20

proc_cnt=0

for dT in "${dTs[@]}"; do
    
    input_file=../01_various_SST_front_wavelength/lab_mysdg_hires_constant_diff/case_dT${dT}_wid050_wpkt01/wrfout_d01_0001-01-01_00\:00\:00

    output_dir=$fig_dir/$dT
    mkdir -p $output_dir

    for t in $( seq 0 576 ); do
     
        output_file=$output_dir/$( printf "%03d" $t ).png 
        python3 plot_snapshot.py  \
            --wrfout $input_file  \
            --time-idx $t        \
            --title "dT: $dT, time : $(( $t * 5 )) min" \
            --output $output_file \
            --no-display &

        proc_cnt=$(( $proc_cnt + 1))
        
        if (( $proc_cnt >= $nproc )) ; then
            echo "Max proc reached: $nproc"
            wait
            proc_cnt=0
        fi
    done

done
