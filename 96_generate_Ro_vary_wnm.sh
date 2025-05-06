#!/bin/bash

source 00_setup.sh
source 98_trapkill.sh
nproc=1

time_avg_interval=60   # minutes

hrs_beg=$(( 24 * 5 ))
    
output_dir=$gendata_dir/Ro_analysis

mkdir -p $output_dir


thumbnail_skip=0
for dT in 100 ; do
for bl_scheme in MYNN25 MYJ YSU ; do
#for bl_scheme in MYJ ; do
for target_lab in lab_SIMPLE lab_FULL ; do

    dhr=$( get_dhr $bl_scheme ) 
    hrs_end=$(( $hrs_beg + $dhr ))

    output_dir=$fig_dir/coherence_analysis
    
    mkdir -p $output_dir
    
    input_dirs_base=""
    input_dirs=""
    dSSTs=""
    tracking_wnms=""
    for Ug in 20 ; do
    for wnm in 004 005 007 010 020 040; do
    #for wnm in 004 040; do
    #for wnm in 004 010 ; do
    #for wnm in 004 005 ; do
    
        if [[ "$target_lab" =~ "FULL" ]]; then
            mph=on
        elif [[ "$target_lab" =~ "SIMPLE" ]]; then
            mph=off
        fi

        casename="case_mph-${mph}_wnm${wnm}_U${Ug}_dT${dT}_${bl_scheme}"
        casename_base="case_mph-${mph}_wnm000_U${Ug}_dT000_${bl_scheme}"

        input_dir_root=$gendata_dir/preavg/$target_lab
        
        input_dir="$input_dir_root/$casename"
        input_dirs="$input_dirs $input_dir"

        input_dir_base="$input_dir_root/$casename_base"
        input_dirs_base="$input_dirs_base $input_dir_base"

        tracking_wnms="$tracking_wnms $wnm"

    done
    done
        
    output_file=$gendata_dir/Ro_analysis/Ro_analysis_vary_wnm_${target_lab}_dSST${dT}_${bl_scheme}_hr${hrs_beg}-${hrs_end}.nc

    if [ -f "$output_file" ] ; then

        echo "File $output_file already exists. Skip."

    else

        eval "python3 src/gen_Ro_vary_wnm.py    \
            --input-dirs $input_dirs                   \
            --input-dirs-base $input_dirs_base         \
            --output $output_file                      \
            --time-rng $hrs_beg $hrs_end               \
            --exp-beg-time '2001-01-01 00:00:00'       \
            --wrfout-data-interval 3600                \
            --frames-per-wrfout-file 12                \
            --tracking-wnms ${tracking_wnms[@]}        
        " &

        thumbnail_skip=$(( $thumbnail_skip + 1 ))

        nproc_cnt=$(( $nproc_cnt + 1 ))
        if (( $nproc_cnt >= $nproc )) ; then
            echo "Max batch_cnt reached: $nproc"
            wait
            nproc_cnt=0
        fi

    fi
done
done
done

wait
echo "Done."
