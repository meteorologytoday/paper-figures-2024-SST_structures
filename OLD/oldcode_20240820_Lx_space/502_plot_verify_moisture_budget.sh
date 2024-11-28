#!/bin/bash

source 00_setup.sh


nproc=1
output_fig_dir=$fig_dir/lab_verify_moisture_budget_V4.5.2/timeseries

for _dir in $output_fig_dir $cache_dir ; do
    echo "mkdir -p $_dir"
    mkdir -p $_dir
done


offset=$(( 24 * 0 ))
dhr=$(( 24 * 9 ))



#dhr=6
trap "exit" INT TERM
trap "echo 'Exiting... ready to kill jobs... '; kill 0" EXIT
analysis_root=data_verify_moisture/analysis_timeseries


PREFIX=WRF452-


for smooth in 1 25 ; do
for Lx in 002 ; do


    input_dirs=(
        $analysis_root/lab_verify_moisture_budget_V4.5.2/case_mph-on_Lx${Lx}_U20_dT000_MYNN25/avg_before_analysis-TRUE
        $analysis_root/lab_verify_moisture_budget_V4.5.2/lab_verify_moisture_budget_V4.5.2/case_mph-on_Lx${Lx}_U20_dT000_YSU/avg_before_analysis-TRUE
        $analysis_root/lab_verify_moisture_budget_V4.5.2/case_mph-on_Lx${Lx}_U20_dT000_MYJ/avg_before_analysis-TRUE
    )


    linestyles=(
        "solid"
        "solid"
        "solid"
    )

    linecolors=(
        "black"
        "red"
        "blue"
    )

    labels=(
        "MYNN25"
        "YSU"
        "MYJ"
    )


    hrs_beg=$( printf "%03d" $(( $offset )) )
    hrs_end=$( printf "%03d" $(( $offset + $dhr )) )

    echo "Doing diagnostic AUX"
    output="$output_fig_dir/${PREFIX}AUX_Lx${Lx}_timeseries_smooth-${smooth}_${hrs_beg}-${hrs_end}.png"
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
        --ncols 2                            \
        --smooth $smooth                     \
        --no-display                         \
        --show-labels                        \
        --varnames    QVAPOR_TTL        PRECIP              \
                      QFX               WATER_BUDGET_RES    \
        --output $output & 

#        --varnames    QVAPOR_TTL QCLOUD_TTL QRAIN_TTL  THETA_MEAN \
#                      TKE_TTL    WATER_BUDGET_RES     BLANK      BLANK      \

    if [ ] ; then
    echo "Doing diagnostic simple"
    output="$output_fig_dir/SIMPLE_Lx${Lx}_timeseries_smooth-${smooth}_${hrs_beg}-${hrs_end}.svg"
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
        --ncols 4                            \
        --smooth $smooth                     \
        --no-display                         \
        --varnames    PBLH TA QA WND_m PRECIP HFX LH \
        --output $output & 



    echo "Doing diagnostic HEATFLX"
    output="$output_fig_dir/HEATFLX_Lx${Lx}_timeseries_smooth-${smooth}_${hrs_beg}-${hrs_end}.png"
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
        --smooth $smooth                     \
        --no-display                         \
        --varnames      HFX     C_H_WND_TOA  WND_TOA_cx_mul_C_H  C_H_TOA_cx_mul_WND  C_H_WND_cx_mul_TOA  \
                        LH      C_Q_WND_QOA  WND_QOA_cx_mul_C_Q  C_Q_QOA_cx_mul_WND  C_Q_WND_cx_mul_QOA  \
        --output $output &

    fi
done
done

wait

echo "PLOTTING DONE."
