#!/bin/bash

source 00_setup.sh

nproc=1
output_fig_dir=$fig_dir/timeseries

for _dir in $output_fig_dir $cache_dir ; do
    echo "mkdir -p $_dir"
    mkdir -p $_dir
done


offset=$(( 24 * 0 ))
dhr=$(( 24 * 15 ))



#dhr=6
trap "exit" INT TERM
trap "echo 'Exiting... ready to kill jobs... '; kill 0" EXIT
analysis_root=$gendata_dir/analysis_timeseries


for Lx in 100 500 ; do


    input_dirs=(
        $analysis_root/lab_sine_dry/case_mph-off_Lx${Lx}_U20_dT000_MYNN25/avg_before_analysis-TRUE
        $analysis_root/lab_sine_dry/case_mph-off_Lx${Lx}_U20_dT300_MYNN25/avg_before_analysis-TRUE
        $analysis_root/lab_sine_semiwet/case_mph-off_Lx${Lx}_U20_dT000_MYNN25/avg_before_analysis-TRUE
        $analysis_root/lab_sine_semiwet/case_mph-off_Lx${Lx}_U20_dT300_MYNN25/avg_before_analysis-TRUE
        $analysis_root/lab_sine_wet/case_mph-on_Lx${Lx}_U20_dT000_MYNN25/avg_before_analysis-TRUE
        $analysis_root/lab_sine_wet/case_mph-on_Lx${Lx}_U20_dT100_MYNN25/avg_before_analysis-TRUE
        $analysis_root/lab_sine_wet/case_mph-on_Lx${Lx}_U20_dT300_MYNN25/avg_before_analysis-TRUE
    )


    linestyles=(
        "dashed"
        "solid"
        "dashed"
        "solid"
        "dashed"
        "dotted"
        "solid"
    )

    linecolors=(
        "gray"
        "gray"
        "red"
        "red"
        "blue"
        "blue"
        "blue"
    )

    labels=(
        "DRY-000"
        "DRY-300"
        "SEMIWET-000"
        "SEMIWET-300"
        "WET-000"
        "WET-100"
        "WET-300"
    )




    hrs_beg=$( printf "%03d" $(( $offset )) )
    hrs_end=$( printf "%03d" $(( $offset + $dhr )) )

    echo "Doing diagnostic simple"
    output="$output_fig_dir/SIMPLE_Lx${Lx}_timeseries_${hrs_beg}-${hrs_end}.png"
    extra_title=""
    python3 src/plot_timeseries_see_steady_state.py \
        --input-dirs "${input_dirs[@]}"      \
        --linestyles "${linestyles[@]}"      \
        --linecolors "${linecolors[@]}"      \
        --labels "${labels[@]}"              \
        --exp-beg-time "2001-01-01 00:00:00" \
        --wrfout-data-interval 3600          \
        --frames-per-wrfout-file 12          \
        --time-rng $hrs_beg $hrs_end         \
        --extra-title "$extra_title"         \
        --tick-interval-hour 24              \
        --ncols 3                            \
        --no-display                         \
        --varnames    PBLH TA QA PRECIP HFX LH \
        --output $output & 



    echo "Doing diagnostic HEATFLX"
    output="$output_fig_dir/HEATFLX_Lx${Lx}_timeseries_${hrs_beg}-${hrs_end}.png"
    extra_title=""
    python3 src/plot_timeseries_see_steady_state.py \
        --input-dirs "${input_dirs[@]}"      \
        --linestyles "${linestyles[@]}"      \
        --linecolors "${linecolors[@]}"      \
        --labels "${labels[@]}"              \
        --exp-beg-time "2001-01-01 00:00:00" \
        --wrfout-data-interval 3600          \
        --frames-per-wrfout-file 12          \
        --time-rng $hrs_beg $hrs_end         \
        --extra-title "$extra_title"         \
        --tick-interval-hour 24              \
        --ncols 5                            \
        --no-display                         \
        --varnames      HFX     C_H_WND_TOA  WND_TOA_cx_mul_C_H  C_H_TOA_cx_mul_WND  C_H_WND_cx_mul_TOA  \
                        LH      C_Q_WND_QOA  WND_QOA_cx_mul_C_Q  C_Q_QOA_cx_mul_WND  C_Q_WND_cx_mul_QOA  \
        --output $output &


    wait

done
