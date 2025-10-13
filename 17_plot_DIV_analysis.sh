#!/bin/bash

source 00_setup.sh
    
how_many_cycles_to_plot=2

beg_days=(
    5
    10
)



nproc=20

proc_cnt=0

target_labs=(
    lab_FULL
    lab_SIMPLE
)

bl_schemes=(
    MYNN25
    MYJ
    YSU
)

wnms=(
    010
    005
)


source 98_trapkill.sh

for xmavg_half_window_size in 10 ; do
for dT in 100; do
for wnm in ${wnms[@]} ; do
for U in "${Us[@]}" ; do
for target_lab in ${target_labs[@]} ; do
for bl_scheme in ${bl_schemes[@]} ; do
 
    thumbnail_skip=0

    dhr=$( get_dhr $bl_scheme ) 
    output_fig_dir=$fig_dir/div_analysis_dhr-${dhr}

    if [[ "$bl_scheme" = "MYNN25" ]]; then
        tke_analysis=TRUE 
    elif [[ "$bl_scheme" = "YSU" ]]; then
        tke_analysis=FALSE 
    elif [[ "$bl_scheme" = "MYJ" ]]; then
        tke_analysis=FALSE 
    fi 

    if [[ "$target_lab" =~ "FULL" ]]; then
        exp_name="FULL"
    elif [[ "$target_lab" =~ "SIMPLE" ]]; then
        exp_name="SIMPLE"
    fi

    if [[ "$target_lab" =~ "FULL" ]]; then
        mph=on
    elif [[ "$target_lab" =~ "SIMPLE" ]]; then
        mph=off
    fi
    
    if [[ "$wnm" = "005" ]]; then
        thumbnail_skip=2
    fi 

    exp_name="${exp_name}."

    preavg_dir=$( gen_preavg_dir $U )

    input_dir=$preavg_dir/$target_lab/case_mph-${mph}_wnm${wnm}_U${U}_dT${dT}_${bl_scheme}
    output_dir=$output_fig_dir/$target_lab/case_mph-${mph}_wnm${wnm}_U${U}_dT${dT}_${bl_scheme}
    
    input_dir_base=$preavg_dir/$target_lab/case_mph-${mph}_wnm000_U${U}_dT000_${bl_scheme}

    mkdir -p $output_dir

    x_rng=$( python3 -c "print(\"%.3f\" % ( $how_many_cycles_to_plot * $Lx / float('${wnm}'.lstrip('0')) , ))" )


    for beg_day in "${beg_days[@]}"; do
     
        hrs_beg=$( printf "%02d" $(( $beg_day * 24 )) )
        hrs_end=$( printf "%02d" $(( $hrs_beg + $dhr )) )

        output_name="$output_dir/div_analysis_${hrs_beg}-${hrs_end}_halfwindow-${xmavg_half_window_size}.${fig_ext}"
        extra_title="${exp_name}${bl_scheme}."

#            --input-dir-base $input_dir_base  \
        eval "python3 src/plot_divergence_analysis.py  \
            --input-dir $input_dir  \
            --exp-beg-time '2001-01-01 00:00:00' \
            --wrfout-data-interval 3600          \
            --frames-per-wrfout-file 12          \
            --time-rng $(( $hrs_beg * 60 )) $(( $hrs_end * 60 ))  \
            --extra-title '$extra_title'         \
            --x-rng 0 $x_rng        \
            --f0 $f0 \
            --xmavg-half-window-size ${xmavg_half_window_size} \
            --output $output_name \
            --thumbnail-skip $thumbnail_skip \
            --height-mode Z \
            --heights 100 \
            --no-display 
        " &



        proc_cnt=$(( $proc_cnt + 1))
        
        if (( $proc_cnt >= $nproc )) ; then
            echo "Max proc reached: $nproc"
            wait
            proc_cnt=0
        fi
    done
done
done
done
done
done
done
wait
