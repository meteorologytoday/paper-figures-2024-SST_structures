#!/bin/bash

source 00_setup.sh

nproc=2
output_fig_dir=$fig_dir/timeseries
trap "exit" INT TERM
trap "echo 'Exiting... ready to kill jobs... '; kill 0" EXIT

analysis_root=$gendata_dir/analysis_timeseries
output_dir_root=$gendata_dir/PBLH_detect

hrs_beg=0
hrs_end=240

for avg_before_analysis in "TRUE" ; do
for target_lab in lab_sine_wet ; do                
    #for Lx in "020" "040" "060" "080" "100" "120" "140" "160" "180" "200" "250" "300" "350" "400" "500" ; do
    for Lx in "020" "040" ; do
        for U in "20" ; do
            for _bl_scheme in "MYNN25" ; do
                #for dT in 000 020 040 060 080 100 150 200 250 300 ; do
                for dT in 100 ; do
                  
                    if [[ "$target_lab" =~ "wet" ]]; then
                        mph=on
                    elif [[ "$target_lab" =~ "dry" ]]; then
                        mph=off
                    fi

                    casename="case_mph-${mph}_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}"
                    input_dir="$analysis_root/$target_lab/$casename/avg_before_analysis-${avg_before_analysis}"
                    output_dir=$output_dir_root/$target_lab/avg_before_analysis-${avg_before_analysis}
                    output_file=$output_dir/${casename}.nc

                    ((j=j%nproc)); ((j++==0)) && wait

                    python3 src/detect_PBLH_time.py      \
                        --input-dir $input_dir                  \
                        --output $output_file                   \
                        --exp-beg-time "2001-01-01 00:00:00"    \
                        --time-rng $hrs_beg $hrs_end            \
                        --wrfout-data-interval 3600             \
                        --frames-per-wrfout-file 1             \
                        --PBLH-rng 750 850       &

                done
            done
        done
    done
done
done

