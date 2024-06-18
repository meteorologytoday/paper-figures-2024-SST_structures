#!/bin/bash

source 00_setup.sh

input_dirs=""
exp_names=""

output_dir_root=$gendata_dir/analysis_timeseries

hrs_beg=$(( 24 * 0 ))
hrs_end=$(( 24 * 16 ))


time_avg_interval=60   # minutes

batch_cnt_limit=3
nproc=20

#batch_cnt_limit=3
#nproc=4



trap "exit" INT TERM
trap "echo 'Exiting... ready to kill jobs... '; kill 0" EXIT

for avg_before_analysis in "TRUE" ; do
for _bl_scheme in MYJ YSU MYNN25  ; do
for target_lab in  lab_sine_WETLWSW lab_sine_WETLW lab_sine_WET lab_sine_DRY ; do 
#for target_lab in  lab_sine_WET ; do 
for Lx in 500 400 300 200 100 050 ; do
for dT in 000 050 100 150 200 250 300 ; do
#for dT in 300 ; do
for U in 20 ; do

    if [[ "$target_lab" =~ "SEMIWET" ]]; then
        mph=off
    elif [[ "$target_lab" =~ "WET" ]]; then
        mph=on
    elif [[ "$target_lab" =~ "DRY" ]]; then
        mph=off
    fi

    casename="case_mph-${mph}_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}"
    input_dir_root=$data_sim_dir/$target_lab
    input_dir="${input_dir_root}/${casename}"
    output_dir="$output_dir_root/$target_lab/$casename/avg_before_analysis-${avg_before_analysis}"

    mkdir -p $output_dir


    python3 src/gen_analysis_timeseries_parallel.py      \
        --input-dir $input_dir                     \
        --output-dir $output_dir                   \
        --exp-beg-time "2001-01-01 00:00:00"       \
        --time-rng $hrs_beg $hrs_end               \
        --time-avg-interval $time_avg_interval     \
        --avg-before-analysis $avg_before_analysis \
        --x-rng 0 $Lx                              \
        --wrfout-data-interval 60                  \
        --frames-per-wrfout-file 60                \
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
done

wait

echo "Done"
