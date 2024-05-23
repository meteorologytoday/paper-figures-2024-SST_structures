#!/bin/bash

source 00_setup.sh

input_dirs=""
exp_names=""

output_dir_root=$gendata_dir/analysis_timeseries

hrs_beg=0
hrs_end=$(( 24 * 20 )) 


time_avg_interval=60   # minutes


N=10

trap "exit" INT TERM
trap "echo 'Exiting... ready to kill jobs... '; kill 0" EXIT

for avg_before_analysis in "TRUE" ; do
for target_lab in lab_sine_wet lab_sine_dry ; do                
#    for Lx in "020" "500" "100" ; do
#        for U in "10" "20" ; do
#            for _bl_scheme in "MYNN25" ; do
#                for dT in 020 300 ; do

    for Lx in "100" ; do
        for U in "20" ; do
            for _bl_scheme in "MYNN25" ; do
                for dT in 000 ; do
                  
                    if [[ "$target_lab" =~ "wet" ]]; then
                        mph=on
                    elif [[ "$target_lab" =~ "dry" ]]; then
                        mph=off
                    fi

                    casename="case_mph-${mph}_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}"
                    input_dir_root=$data_dir/$target_lab
                    input_dir="${input_dir_root}/${casename}"
                    output_dir="$output_dir_root/$target_lab/$casename/avg_before_analysis-${avg_before_analysis}"

                    mkdir -p $output_dir

                    ((j=j%N)); ((j++==0)) && wait

                    python3 src/gen_analysis_timeseries.py      \
                        --input-dir $input_dir                  \
                        --output-dir $output_dir                \
                        --exp-beg-time "2001-01-01 00:00:00"    \
                        --time-rng $hrs_beg $hrs_end            \
                        --time-avg-interval $time_avg_interval  \
                        --avg-before-analysis $avg_before_analysis \
                        --x-rng 0 $Lx                           \
                        --wrfout-data-interval 60               \
                        --frames-per-wrfout-file 60 &

                done
            done
        done
    done
done
done

wait

echo "Done"
