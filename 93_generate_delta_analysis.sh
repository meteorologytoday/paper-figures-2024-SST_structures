#!/bin/bash

source 00_setup.sh

input_dirs=""
exp_names=""



hrs_beg=$(( 24 * 5 ))
hrs_end=$(( 24 * 11 ))


time_avg_interval=60   # minutes

batch_cnt_limit=3
nproc=10

#batch_cnt_limit=1
#nproc=1



trap "exit" INT TERM
trap "echo 'Exiting... ready to kill jobs... '; kill 0" EXIT

for analysis_style in STYLE1 ; do


    output_dir_root=$gendata_dir/delta_analysis_style-$analysis_style


    for avg_before_analysis in "TRUE" ; do
    #for _bl_scheme in MYNN25  MYJ YSU ; do
    for _bl_scheme in MYNN25 MYJ YSU ; do
    #for _bl_scheme in YSU ; do
    for target_lab in  lab_FIXEDDOMAIN_SST_sine_WETLWSW ; do #lab_FIXEDDOMAIN_SST_sine_DRY; do 

    for wnm in 000 004 005 007 010 020 040 ; do

        if [ "$wnm" = "004" ] || [ "$wnm" = "010" ]; then

            dTs=( 300 000 050 100 150 200 250 300 )

        else
            dTs=( 000 100 300 )
        fi
            

    for dT in "${dTs[@]}" ; do
    for U in 20 ; do

        if [[ "$target_lab" =~ "SEMIWET" ]]; then
            mph=off
        elif [[ "$target_lab" =~ "WET" ]]; then
            mph=on
        elif [[ "$target_lab" =~ "DRY" ]]; then
            mph=off
        fi

        casename="case_mph-${mph}_wnm${wnm}_U${U}_dT${dT}_${_bl_scheme}"
        casename_base="case_mph-${mph}_wnm000_U${U}_dT000_${_bl_scheme}"
        input_dir_root=$preavg_dir/$target_lab

        input_dir="${input_dir_root}/${casename}"
        input_dir_base="${input_dir_root}/${casename_base}"

        output_dir="$output_dir_root/$target_lab/$casename/avg_before_analysis-${avg_before_analysis}"

        mkdir -p $output_dir


        python3 src/gen_delta_analysis.py              \
            --input-dir $input_dir                     \
            --input-dir-base $input_dir_base           \
            --output-dir $output_dir                   \
            --analysis-style $analysis_style           \
            --exp-beg-time "2001-01-01 00:00:00"       \
            --time-rng $hrs_beg $hrs_end               \
            --time-avg-interval $time_avg_interval     \
            --avg-before-analysis $avg_before_analysis \
            --wrfout-data-interval 3600                \
            --frames-per-wrfout-file 12                \
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
done

wait

echo "Done"
