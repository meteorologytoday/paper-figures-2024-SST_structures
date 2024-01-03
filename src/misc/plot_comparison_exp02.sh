#!/bin/bash

fig_dir=figures_exp02_compare_ML_and_bl_scheme

plot_infos=(
    
)

input_dirs=""
input_dir_root="../12_SST_front/lab_front"
for dT in -300 -100 000 100 300 ; do
    for wid in 050 ; do
        for MLsetup in woML ; do
            for MLscheme in YSU ; do 
                input_dirs="$input_dirs $input_dir_root/case_dT${dT}_wid${wid}_${MLsetup}_${MLscheme}"
            done
        done
    done
done

output_dir=$fig_dir
mkdir -p $output_dir
avg_len=2
t=50
python3 plot_comparison.py  \
    --input-dirs $input_dirs  \
    --time-rng "0001-01-02 00:00:00" "0001-01-02 03:00:00" \
    --title "$title" 

