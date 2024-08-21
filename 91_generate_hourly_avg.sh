#!/bin/bash

source 00_setup.sh

input_dirs=""
exp_names=""

output_dir_root=$preavg_dir

hrs_beg=$(( 24 * 5 ))
hrs_end=$(( 24 * 11 ))


time_avg_interval=60   # minutes

batch_cnt_limit=2
nproc=10

#batch_cnt_limit=1
#nproc=1


trap "exit" INT TERM
trap "echo 'Exiting... ready to kill jobs... '; kill 0" EXIT

#for _bl_scheme in MYJ YSU MYNN25 ; do
for _bl_scheme in MYNN25 ; do

#for target_lab in lab_FIXEDDOMAIN_SST_sine_WETLWSW lab_FIXEDDOMAIN_SST_sine_DRY ; do 
for target_lab in lab_FIXEDDOMAIN_SST_sine_DRY ; do 
#for target_lab in lab_sine_WETLWSW ; do 
for wnm in 000 010 004 005 007 010 020 040 ; do
for dT in 000 300 050 100 150 200 250 ; do
for U in 20 ; do

    if [[ "$target_lab" =~ "SEMIWET" ]]; then
        mph=off
    elif [[ "$target_lab" =~ "WET" ]]; then
        mph=on
    elif [[ "$target_lab" =~ "DRY" ]]; then
        mph=off
    fi

    casename="case_mph-${mph}_wnm${wnm}_U${U}_dT${dT}_${_bl_scheme}"
    input_dir_root=$data_sim_dir/$target_lab
    input_dir="${input_dir_root}/${casename}"
    output_dir="$output_dir_root/$target_lab/$casename"

    mkdir -p $output_dir


    python3 src/preavg.py \
        --input-dir $input_dir                     \
        --output-dir $output_dir                   \
        --exp-beg-time "2001-01-01 00:00:00"       \
        --time-rng $hrs_beg $hrs_end               \
        --time-avg-interval $time_avg_interval     \
        --wrfout-data-interval 600                 \
        --frames-per-wrfout-file 36                \
        --output-count 12                          \
        --nproc $nproc &


    batch_cnt=$(( $batch_cnt + 1))
    
    if (( $batch_cnt >= $batch_cnt_limit )) ; then
        echo "Max batch_cnt reached: $batch_cnt"
        wait
        batch_cnt=0
    fi
   
done
done
done
done
done

wait

echo "Done"
