#!/bin/bash

source 00_setup.sh

input_dirs=""
exp_names=""



hrs_beg=$(( 24 * 5 ))
hrs_end=$(( 24 * 16 ))


time_avg_interval=60   # minutes

batch_cnt_limit=1
nproc=10

source 98_trapkill.sh

for analysis_style in STYLE1 ; do 




    for avg_before_analysis in "TRUE" ; do
    for _bl_scheme in MYNN25 YSU MYJ ; do
    for target_lab in  lab_FULL lab_SIMPLE ; do
    for U in "${Us[@]}" ; do
    for wnm in 004 005 007 010 020 040 000 ; do
    for dT in  000 010 030 050 100 150 200 250 300 ; do


        if_wanted=$( python3 src/check_if_simulation_wanted.py --target-lab $target_lab --dT100 $dT --U $U --wnm $wnm )
        label="${target_lab}-U${U}-wnm${wnm}-dT${dT}_${_bl_scheme}"

        if [ "$if_wanted" = "TRUE" ]; then
            echo "Need $label"
        elif [ "$if_wanted" = "FALSE" ]; then
            echo "Skip $label"
            continue
        else
            echo "Unknown answer $if_wanted. Skip $label"
            continue
        fi 

        if [[ "$target_lab" =~ "FULL" ]]; then
            mph=on
        elif [[ "$target_lab" =~ "SIMPLE" ]]; then
            mph=off
        fi

        gendata_dir=$( gen_gendata_dir $U )
        preavg_dir=$( gen_preavg_dir $U )
        output_dir_root=$( gen_delta_analysis_dir $U $analysis_style )

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
