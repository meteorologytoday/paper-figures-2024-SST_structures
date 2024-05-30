#!/bin/bash

source 00_setup.sh

analysis_root=$gendata_dir/analysis_timeseries

stat_infos=(
    $(( 12 * 24 ))    $(( 3 * 24 ))
    $((  2 * 24 ))    $(( 3 * 24 ))
    $((  1 * 24 ))    $(( 3 * 24 ))
)


nparams=2
for (( i=0 ; i < $(( ${#stat_infos[@]} / $nparams )) ; i++ )); do

    offset="${stat_infos[$(( i * $nparams + 0 ))]}"
    dhr="${stat_infos[$(( i * $nparams + 1 ))]}"

    for Lx in 500 ; do

        echo "Doing Lx = $Lx"

        input_dirs=(
            $analysis_root/lab_sine_wet/case_mph-on_Lx${Lx}_U20_dT000_MYNN25/avg_before_analysis-TRUE
            $analysis_root/lab_sine_wet/case_mph-on_Lx${Lx}_U20_dT100_MYNN25/avg_before_analysis-TRUE
            $analysis_root/lab_sine_wet/case_mph-on_Lx${Lx}_U20_dT300_MYNN25/avg_before_analysis-TRUE
        )


        labels=(
            "WET-000"
            "WET-100"
            "WET-300"
        )




        hrs_beg=$( printf "%03d" $(( $offset )) )
        hrs_end=$( printf "%03d" $(( $offset + $dhr )) )

        echo "Doing diagnostic simple"
        python3 src/timeseries_stat.py \
            --input-dirs "${input_dirs[@]}"      \
            --labels "${labels[@]}"              \
            --exp-beg-time "2001-01-01 00:00:00" \
            --wrfout-data-interval 3600          \
            --frames-per-wrfout-file 12          \
            --time-rng $hrs_beg $hrs_end         \
            --varnames      HFX     C_H_WND_TOA  WND_TOA_cx_mul_C_H  C_H_TOA_cx_mul_WND  C_H_WND_cx_mul_TOA  \
                            LH      C_Q_WND_QOA  WND_QOA_cx_mul_C_Q  C_Q_QOA_cx_mul_WND  C_Q_WND_cx_mul_QOA  &


        wait

    done

done
