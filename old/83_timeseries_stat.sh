#!/bin/bash

source 00_setup.sh

analysis_root=$gendata_dir/analysis_timeseries

stat_infos=(
    $((  0 * 24 ))    $(( 5 * 24 ))
    $((  5 * 24 ))    $(( 5 * 24 ))
    $((  10 * 24 ))   $(( 5 * 24 ))
)


nparams=2
for (( i=0 ; i < $(( ${#stat_infos[@]} / $nparams )) ; i++ )); do

    offset="${stat_infos[$(( i * $nparams + 0 ))]}"
    dhr="${stat_infos[$(( i * $nparams + 1 ))]}"

    for target_lab in lab_sine_dry lab_sine_wetlwsw ; do

        if [[ "$target_lab" =~ "semiwet" ]]; then
            mph=off
        elif [[ "$target_lab" =~ "wet" ]]; then
            mph=on
        elif [[ "$target_lab" =~ "dry" ]]; then
            mph=off
        fi



        for Lx in 500 ; do
        for Ug in 20; do
        for dT in 000 300 ; do

            echo "Doing Lx=$Lx , dT=$dT"

            input_dir=$analysis_root/$target_lab/case_mph-${mph}_Lx${Lx}_U${Ug}_dT${dT}_MYNN25/avg_before_analysis-TRUE


            hrs_beg=$( printf "%03d" $(( $offset )) )
            hrs_end=$( printf "%03d" $(( $offset + $dhr )) )

            output_dir=$gendata_dir/timeseries_stat/$target_lab
            output=$output_dir/Lx${Lx}_U${Ug}_dT${dT}_${hrs_beg}-${hrs_end}.csv

            mkdir -p ${output_dir}
            
            echo "Doing diagnostic simple"
            python3 src/timeseries_stat_new.py \
                --input-dir $input_dir      \
                --exp-beg-time "2001-01-01 00:00:00" \
                --wrfout-data-interval 3600          \
                --frames-per-wrfout-file 12          \
                --time-rng $hrs_beg $hrs_end         \
                --varnames      HFX     C_H_WND_TOA  WND_TOA_cx_mul_C_H  C_H_TOA_cx_mul_WND  C_H_WND_cx_mul_TOA  \
                                LH      C_Q_WND_QOA  WND_QOA_cx_mul_C_Q  C_Q_QOA_cx_mul_WND  C_Q_WND_cx_mul_QOA  \
                                PRECIP \
                --output ${output}


            wait

        done
        done
        done
    done
done
