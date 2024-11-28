#!/bin/bash

source 00_setup.sh

nproc=1
output_fig_dir=$fig_dir/timeseries_relative_WRFV4.6.0

for _dir in $output_fig_dir $cache_dir ; do
    echo "mkdir -p $_dir"
    mkdir -p $_dir
done


offset=$(( 24 * 0 ))
dhr=$(( 24 * 15 ))


source 98_trapkill.sh

#dhr=6
#trap "exit" INT TERM
#trap "echo 'Exiting... ready to kill jobs... '; kill 0" EXIT
analysis_root=$gendata_dir/analysis_timeseries


for avg in TRUE ; do
for bl_scheme in MYJ MYNN25 YSU ; do
for smooth in 25 49 ; do
for Lx in 100 500 ; do


    input_dirs=(
        $analysis_root/lab_sine_DRY/case_mph-off_Lx${Lx}_U20_dT300_${bl_scheme}/avg_before_analysis-${avg}
#        $analysis_root/lab_sine_WET/case_mph-on_Lx${Lx}_U20_dT300_${bl_scheme}/avg_before_analysis-${avg}
#        $analysis_root/lab_sine_WETLW/case_mph-on_Lx${Lx}_U20_dT300_${bl_scheme}/avg_before_analysis-${avg}
        $analysis_root/lab_sine_WETLWSW/case_mph-on_Lx${Lx}_U20_dT300_${bl_scheme}/avg_before_analysis-${avg}
    )

    input_dirs_base=(
        $analysis_root/lab_sine_DRY/case_mph-off_Lx${Lx}_U20_dT000_${bl_scheme}/avg_before_analysis-TRUE
#        $analysis_root/lab_sine_WET/case_mph-on_Lx${Lx}_U20_dT000_${bl_scheme}/avg_before_analysis-TRUE
#        $analysis_root/lab_sine_WETLW/case_mph-on_Lx${Lx}_U20_dT000_${bl_scheme}/avg_before_analysis-TRUE
        $analysis_root/lab_sine_WETLWSW/case_mph-on_Lx${Lx}_U20_dT000_${bl_scheme}/avg_before_analysis-TRUE
    )



    linestyles=(
        "solid"
        "solid"
        "solid"
        "solid"
    )

    linecolors=(
        "red"
        "black"
#        "gray"
#        "orangered"
#        "dodgerblue"
#        "magenta"
    )

    labels=(
        "DRY"
#        "WET"
#        "WET-LW"
        "WET-LWSW"
    )


    hrs_beg=$( printf "%03d" $(( $offset )) )
    hrs_end=$( printf "%03d" $(( $offset + $dhr )) )


    echo "Doing diagnostic HEATFLX"
    output="$output_fig_dir/HEATFLX_avg-${avg}_Lx${Lx}_${bl_scheme}_timeseries_smooth-${smooth}_${hrs_beg}-${hrs_end}.svg"
    extra_title=""
    python3 src/plot_timeseries_see_steady_state.py \
        --input-dirs "${input_dirs[@]}"      \
        --input-dirs-base "${input_dirs_base[@]}"      \
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

        #                QVAPOR_TTL IVT  THETA_MEAN SWDOWN  OLR \


done
done
done
done

wait

echo "PLOTTING DONE."
