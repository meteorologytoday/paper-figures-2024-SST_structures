#!/bin/bash

source 00_setup.sh

nproc=5
avg_interval=60
output_fig_dir=$fig_dir/snapshots_vertical_timeseries_WRFV4.6.0_avg-${avg_interval}

tmp_dir=$gendata_dir/vertical_profile_tmp


for _dir in $output_fig_dir $cache_dir ; do
    echo "mkdir -p $_dir"
    mkdir -p $_dir
done


offset=$(( 24 * 10 ))
dhr=$(( 24 * 5 ))

#dhr=$(( 1 ))


source 98_trapkill.sh

#dhr=6
#trap "exit" INT TERM
#trap "echo 'Exiting... ready to kill jobs... '; kill 0" EXIT

for offset in 240 0 120 ; do 


for bl_scheme in MYJ YSU MYNN25; do
for avg in TRUE ; do
for Lx in 500 ; do


    input_dirs=(
#        $data_sim_dir/lab_sine_DRY/case_mph-off_Lx${Lx}_U20_dT000_${bl_scheme}
#        $data_sim_dir/lab_sine_DRY/case_mph-off_Lx${Lx}_U20_dT300_${bl_scheme}
        $data_sim_dir/lab_sine_WETLWSW/case_mph-on_Lx${Lx}_U20_dT000_${bl_scheme}
        $data_sim_dir/lab_sine_WETLWSW/case_mph-on_Lx${Lx}_U20_dT300_${bl_scheme}
    )


    linestyles=(
        "dashed"
        "solid"
        "dashed"
        "solid"
    )

    linecolors=(
        "gray"
        "gray"
        "magenta"
        "magenta"
    )

    labels=(
        "DRY-000"
        "DRY-300"
        "WETRAD-000"
        "WETRAD-300"
    )


    hrs_beg=$( printf "%03d" $(( $offset )) )
    hrs_end=$( printf "%03d" $(( $offset + $dhr )) )



    echo "Doing diagnostic simple"
    output="$output_fig_dir/VERTICAL_avg-${avg}_Lx${Lx}_${bl_scheme}_${hrs_beg}-${hrs_end}.svg"
    tmp_file="$output_fig_dir/VERTICAL_avg-${avg}_Lx${Lx}_${bl_scheme}_${hrs_beg}-${hrs_end}.svg"
    extra_title="${bl_scheme}."
    python3 src/plot_snapshot_vertical_profile.py \
        --input-dirs "${input_dirs[@]}"      \
        --linestyles "${linestyles[@]}"      \
        --linecolors "${linecolors[@]}"      \
        --labels "${labels[@]}"              \
        --exp-beg-time "2001-01-01 00:00:00" \
        --wrfout-data-interval 60            \
        --frames-per-wrfout-file 60          \
        --time-rng $(( $hrs_beg * 60 )) $(( $hrs_end * 60 ))  \
        --extra-title "$extra_title"         \
        --no-display                         \
        --z-rng 0 5000                       \
        --avg-interval $(( $avg_interval ))  \
        --varnames   QVAPOR T_TOTAL RH QCLOUD \
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
done

wait

echo "PLOTTING DONE."
