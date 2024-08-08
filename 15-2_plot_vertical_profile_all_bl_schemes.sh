#!/bin/bash

source 00_setup.sh

nproc=10
avg_interval=60
output_fig_dir=$fig_dir/snapshots_vertical_timeseries_WRFV4.6.0_avg-${avg_interval}

tmp_dir=$gendata_dir/vertical_profile_tmp


for _dir in $output_fig_dir $cache_dir ; do
    echo "mkdir -p $_dir"
    mkdir -p $_dir
done


dhr=$(( 24 * 5 ))
#dhr=$(( 5 ))

#dhr=$(( 1 ))


source 98_trapkill.sh

#dhr=6
#trap "exit" INT TERM
#trap "echo 'Exiting... ready to kill jobs... '; kill 0" EXIT

for offset in 120 ; do 


for avg in TRUE ; do
for Lx in 500 100 ; do
    
    input_dirs_base=(
        $data_sim_dir/lab_sine_WETLWSW/case_mph-on_Lx${Lx}_U20_dT000_MYNN25
        $data_sim_dir/lab_sine_WETLWSW/case_mph-on_Lx${Lx}_U20_dT000_MYJ
        $data_sim_dir/lab_sine_WETLWSW/case_mph-on_Lx${Lx}_U20_dT000_YSU
    )
    
    
    input_dirs=(
        $data_sim_dir/lab_sine_WETLWSW/case_mph-on_Lx${Lx}_U20_dT300_MYNN25
        $data_sim_dir/lab_sine_WETLWSW/case_mph-on_Lx${Lx}_U20_dT300_MYJ
        $data_sim_dir/lab_sine_WETLWSW/case_mph-on_Lx${Lx}_U20_dT300_YSU
    )
    
    linestyles=(
        "solid"
        "solid"
        "solid"
    )

    linecolors=(
        "black"
        "orangered"
        "dodgerblue"
    )

    labels=(
        "MYNN25"
        "MYJ"
        "YSU"
    )


    hrs_beg=$( printf "%03d" $(( $offset )) )
    hrs_end=$( printf "%03d" $(( $offset + $dhr )) )


    echo "Doing diagnostic simple"
    output="$output_fig_dir/VERTICAL_rel_avg-${avg}_Lx${Lx}_${hrs_beg}-${hrs_end}.svg"
    tmp_file="$output_fig_dir/VERTICAL_rel_avg-${avg}_Lx${Lx}_${hrs_beg}-${hrs_end}.svg"
    extra_title="${bl_scheme}."
    python3 src/plot_snapshot_vertical_profile.py \
        --input-dirs "${input_dirs[@]}"      \
        --input-dirs-base "${input_dirs_base[@]}"      \
        --linestyles "${linestyles[@]}"      \
        --linecolors "${linecolors[@]}"      \
        --labels "${labels[@]}"              \
        --exp-beg-time "2001-01-01 00:00:00" \
        --wrfout-data-interval 60            \
        --frames-per-wrfout-file 60          \
        --time-rng $(( $hrs_beg * 60 )) $(( $hrs_end * 60 ))  \
        --extra-title "$extra_title"         \
        --no-display                         \
        --z-rng 0 10000                      \
        --avg-interval $(( $avg_interval ))  \
        --varnames   QVAPOR T_TOTAL          \
        --output $output & 
        
        proc_cnt=$(( $proc_cnt + 1))
        
        if (( $proc_cnt >= $nproc )) ; then
            echo "Max proc reached: $nproc"
            wait
            proc_cnt=0
        fi

#        --varnames   QVAPOR T_TOTAL  RH H_DIABATIC QCLOUD QRAIN EXCH_H WpRHODQVAPORp_mean RHODQVAPOR_vt RHODQVAPOR_vt_ttl \
done
done
done

wait

echo "PLOTTING DONE."
