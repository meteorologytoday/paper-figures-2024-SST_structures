#!/bin/bash

source 00_setup.sh
    
how_many_cycles_to_plot=2


beg_days=(
    5
)

dhr=$(( 24 * 5 ))
output_fig_dir=$fig_dir/snapshots_dhr-${dhr}

nproc=1

proc_cnt=0

target_labs=(
#    lab_FULL
    lab_SIMPLE
)

bl_schemes=(
    MYNN25
#    MYJ
#    YSU
)

wnms=(
    040
    020
    010
    005
    004
)


source 98_trapkill.sh

for dT in 100; do
for wnm in ${wnms[@]} ; do
for U in "20" ; do
for target_lab in ${target_labs[@]} ; do
for _bl_scheme in ${bl_schemes[@]} ; do
 
    thumbnail_skip_part1=0
    thumbnail_skip_part2=0

       

    if [[ "$_bl_scheme" = "MYNN25" ]]; then
        tke_analysis=TRUE 
    elif [[ "$_bl_scheme" = "YSU" ]]; then
        tke_analysis=FALSE 
    elif [[ "$_bl_scheme" = "MYJ" ]]; then
        tke_analysis=FALSE 
    fi 

    if [[ "$target_lab" =~ "FULL" ]]; then
        exp_name="FULL"
    elif [[ "$target_lab" =~ "SIMPLE" ]]; then
        exp_name="SIMPLE"
    fi

    if [[ "$target_lab" =~ "FULL" ]]; then
        mph=on
        #W_levs=( -50 50 11 )
        W_levs=( -2 2 21 )
            
        thumbnail_skip_part1=5
    
        if [[ "$_bl_scheme" = "MYNN25" ]]; then
            thumbnail_skip_part2=6
        else
            thumbnail_skip_part2=4
        fi

    elif [[ "$target_lab" =~ "SIMPLE" ]]; then
        mph=off
        W_levs=( -2 2 21 )
    fi
 
    exp_name="${exp_name}."


    input_dir=$gendata_dir/preavg/$target_lab/case_mph-${mph}_wnm${wnm}_U${U}_dT${dT}_${_bl_scheme}
    output_dir=$output_fig_dir/$target_lab/case_mph-${mph}_wnm${wnm}_U${U}_dT${dT}_${_bl_scheme}
    
    input_dir_base=$gendata_dir/preavg/$target_lab/case_mph-${mph}_wnm000_U${U}_dT000_${_bl_scheme}

    mkdir -p $output_dir

    part1_x_rng=$( python3 -c "print(\"%.3f\" % ( $how_many_cycles_to_plot * $Lx / float('${wnm}'.lstrip('0')) , ))" )


    for beg_day in "${beg_days[@]}"; do
     
        hrs_beg=$( printf "%02d" $(( $beg_day * 24 )) )
        hrs_end=$( printf "%02d" $(( $hrs_beg + $dhr )) )

        output1_name="$output_dir/snapshot-part1_${hrs_beg}-${hrs_end}.${fig_ext}"
        output2_name="$output_dir/snapshot-part2_${hrs_beg}-${hrs_end}.${fig_ext}"
        extra_title=""

        extra_title="${exp_name}${_bl_scheme}."

        if [ "$dT" == 300 ] ; then
            part1_U10_rng=(-3 3)
            part1_U500_rng=(-3 3)
            part1_DIVVOR10_rng=(-6 6)
            part1_DIVVOR500_rng=(-6 6)
            part1_PRECIP_rng=(-1.2 1.2)
            part1_Q_rng=(-3 3)
            part1_SST_rng=(-4 4)
            part1_W_levs=(-8 8 17)
            part1_TKE_levs=(-1 1 11)

            part2_THETA_rng=(-2 10)
            part2_Nfreq2_rng=(-5 5)
            part2_TKE_rng=(-1 1)
            part2_DTKE_rng=(-1 1)
            part2_U_rng=(-1.5 1.5)
            part2_Q_rng=(-1 1)
  
        elif [ "$dT" == 100 ] ; then
            
            part1_U10_rng=(-2 2)
            part1_U500_rng=(-1 1)
            part1_DIVVOR10_rng=(-5 5)
            part1_DIVVOR500_rng=(-2 2)
            part1_PRECIP_rng=(-0.5 0.5)
            part1_Q_rng=(-1 1)
            part1_SST_rng=(-1.5 1.5)
            part1_W_levs=(-1.4 1.4 15)
            part1_TKE_levs=(-1 1 11)

            if [ "_$bl_scheme" == "YSU" ]; then
                part1_U10_rng=(-3.5 3.5)
                part1_DIVVOR10_rng=(-10 10)
                part1_Q_rng=(-2.5 2.5)
            fi

            part2_THETA_rng=(-2 15)
            part2_Nfreq2_rng=(-5 5)
            part2_TKE_rng=(-1 1)
            part2_DTKE_rng=(-0.001 0.001)
            part2_U_rng=(-1.5 1.5)
            part2_Q_rng=(-1.5 1.5)
  
        else

            echo "ERROR: unspecified dT: $dT"
            exit 1

        fi 


        eval "python3 src/plot_system_state_delta.py  \
            --input-dir $input_dir  \
            --input-dir-base $input_dir_base  \
            --exp-beg-time '2001-01-01 00:00:00' \
            --wrfout-data-interval 3600          \
            --frames-per-wrfout-file 12          \
            --time-rng $(( $hrs_beg * 60 )) $(( $hrs_end * 60 ))  \
            --extra-title '$extra_title'         \
            --part1-z-rng 0 5000 \
            --part2-z-rng 0 5000 \
            --part1-U10-rng ${part1_U10_rng[@]} \
            --part1-U500-rng ${part1_U500_rng[@]} \
            --part1-DIVVOR500-rng ${part1_DIVVOR500_rng[@]} \
            --part1-DIVVOR10-rng ${part1_DIVVOR10_rng[@]} \
            --part1-PRECIP-rng ${part1_PRECIP_rng[@]} \
            --part1-Q-rng ${part1_Q_rng[@]} \
            --part1-SST-rng ${part1_SST_rng[@]} \
            --part1-W-levs ${part1_W_levs[@]} \
            --part1-TKE-levs ${part1_TKE_levs[@]} \
            --part2-THETA-rng ${part2_THETA_rng[@]} \
            --part2-Nfreq2-rng ${part2_Nfreq2_rng[@]} \
            --part2-TKE-rng ${part2_TKE_rng[@]} \
            --part2-DTKE-rng ${part2_DTKE_rng[@]} \
            --part2-U-rng ${part2_U_rng[@]} \
            --part2-Q-rng ${part2_Q_rng[@]} \
            --part1-x-rng 0 $part1_x_rng        \
            --part1-x-rolling 11          \
            --output1 $output1_name \
            --output2 $output2_name \
            --thumbnail-skip-part1 $thumbnail_skip_part1 \
            --thumbnail-skip-part2 $thumbnail_skip_part2 \
            --part2-tke-analysis $tke_analysis \
            --plot-part1 --plot-part2 \
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
