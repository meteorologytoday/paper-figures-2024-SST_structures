#!/bin/bash

source 00_setup.sh

source 98_trapkill.sh


output_dir_root=$preavg_dir

hrs_beg=$(( 24 * 0 ))
hrs_end=$(( 24 * 16 ))


time_avg_interval=60   # minutes

batch_cnt_limit=1
nproc=30


#for _bl_scheme in MYNN25 MYJ YSU ; do
for _bl_scheme in MYNN25 ; do
for target_lab in lab_SIMPLE lab_FULL ; do 

#for _bl_scheme in MYJ ; do
#for target_lab in lab_SIMPLE ; do 
#for wnm in 004 ; do
#for dT in 100 ; do

for wnm in 000 010 004 005 007 010 020 040 ; do
#for wnm in 000 ; do
for dT in 000 010 030 050 100 150 200 250 300 ; do
#for dT in 000 ; do
#for U in 20 ; do
for U in 10 ; do


    if [[ "$target_lab" =~ "SIMPLE" ]]; then
        mph=off
    elif [[ "$target_lab" =~ "FULL" ]]; then
        mph=on
    fi

    if [[ "$wnm" = "000" ]] && [[ "$dT" != "000" ]] ; then
        continue
    fi

    if [[ "$wnm" != "000" ]] && [[ "$dT" = "000" ]] ; then
        continue
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
