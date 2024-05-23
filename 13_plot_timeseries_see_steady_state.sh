#!/bin/bash

source 00_setup.sh

nproc=1
output_fig_dir=$fig_dir/timeseries
cache_dir=$gendata_dir/timeseries_see_steady_state

for _dir in $output_fig_dir $cache_dir ; do
    echo "mkdir -p $_dir"
    mkdir -p $_dir
done


offset=0
dhr=480
dhr=$(( 24 * 20 ))



#dhr=6
trap "exit" INT TERM
trap "echo 'Exiting... ready to kill jobs... '; kill 0" EXIT
analysis_root=$gendata_dir/analysis_timeseries

input_dirs=(
    $analysis_root/lab_sine_dry/case_mph-off_Lx100_U20_dT100_MYNN25/avg_before_analysis-FALSE
    $analysis_root/lab_sine_wet/case_mph-on_Lx100_U20_dT100_MYNN25/avg_before_analysis-TRUE
    $analysis_root/lab_sine_wet/case_mph-on_Lx100_U20_dT000_MYNN25/avg_before_analysis-TRUE
)

input_dirs=(
    $analysis_root/lab_sine_dry/case_mph-off_Lx100_U20_dT300_MYNN25/avg_before_analysis-FALSE
    $analysis_root/lab_sine_wet/case_mph-on_Lx100_U20_dT300_MYNN25/avg_before_analysis-TRUE
    $analysis_root/lab_sine_wet/case_mph-on_Lx100_U20_dT000_MYNN25/avg_before_analysis-TRUE
)


linestyles=(
    "solid"
    "solid"
    "dashed"
)

linecolors=(
    "gray"
    "black"
    "red"
)



hrs_beg=$( printf "%03d" $(( $offset )) )
hrs_end=$( printf "%03d" $(( $offset + $dhr )) )

output="$output_fig_dir/timeseries_${hrs_beg}-${hrs_end}.png"
extra_title=""

echo "Running program..."
python3 src/plot_timeseries_see_steady_state.py \
    --input-dirs "${input_dirs[@]}"      \
    --linestyles "${linestyles[@]}"      \
    --linecolors "${linecolors[@]}"      \
    --exp-beg-time "2001-01-01 00:00:00" \
    --wrfout-data-interval 3600          \
    --frames-per-wrfout-file 1           \
    --time-rng $hrs_beg $hrs_end         \
    --extra-title "$extra_title"         \
    --no-display                         \
    --varnames PBLH TOA_m QOA_m HFX LH WND_TOA_cx_mul_C_H  WND_QOA_cx_mul_C_Q C_Q_QOA_cx_mul_WND C_Q_WND_cx_mul_QOA PRECIP \
    --output $output 

