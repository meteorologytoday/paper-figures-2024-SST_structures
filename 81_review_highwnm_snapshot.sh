#!/bin/bash

source 00_setup.sh
how_many_cycles_to_plot=2    
beg_days=(
    10
)

nproc=1

proc_cnt=0

target_labs=(
    lab_FULL
)

bl_schemes=(
    MYNN25
    MYJ
    YSU
)

wnms=(
    010
)


source 98_trapkill.sh

for dT in 100; do
for wnm in ${wnms[@]} ; do
#for U in "${Us[@]}" ; do
for U in 20 ; do
for target_lab in ${target_labs[@]} ; do
for bl_scheme in ${bl_schemes[@]} ; do
 
    preavg_dir=$( gen_preavg_dir $U )
    dhr=1
    output_fig_dir=$fig_dir/cloud_rain_snapshots_dhr-${dhr}

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
 
    exp_name="${exp_name}."

    input_dir=$preavg_dir/$target_lab/case_mph-${mph}_wnm${wnm}_U${U}_dT${dT}_${bl_scheme}
    output_dir=$output_fig_dir/$target_lab/case_mph-${mph}_wnm${wnm}_U${U}_dT${dT}_${bl_scheme}
    
    mkdir -p $output_dir

    x_rng=$( python3 -c "print(\"%.3f\" % ( $how_many_cycles_to_plot * $Lx / float('${wnm}'.lstrip('0')) , ))" )

    for beg_day in "${beg_days[@]}"; do
     
        hrs_beg=$( printf "%02d" $(( $beg_day * 24 )) )
        hrs_end=$( printf "%02d" $(( $hrs_beg + $dhr )) )

        output="$output_dir/cloud_rain_${hrs_beg}-${hrs_end}.${fig_ext}"
        extra_title=""
        extra_title="${exp_name}${bl_scheme}."

        eval "python3 src/plot_snapshot_cloud_rain.py \
            --input-dir $input_dir  \
            --exp-beg-time '2001-01-01 00:00:00' \
            --wrfout-data-interval 3600          \
            --frames-per-wrfout-file 12          \
            --time-rng $(( $hrs_beg * 60 )) $(( $hrs_end * 60 ))  \
            --extra-title '$extra_title'         \
            --z-rng 0 5000 \
            --x-rng 0 $x_rng        \
            --output $output \
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

wait
