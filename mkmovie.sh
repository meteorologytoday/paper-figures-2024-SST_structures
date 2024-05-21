#!/bin/bash


#ffmpeg -framerate 30 -pattern_type glob -i '*.png' \
#  -c:v libx264 -pix_fmt yuv420p out.mp4

#ffmpeg -f concat -i file_list -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p animation_Lx100_U20_dT100_MYNN25.mp4
output_dir=animations
casenames=(
    Lx100_U30_dT100_MYNN25
    Lx100_U10_dT100_MYNN25
    Lx100_U20_dT100_MYNN25
)

mkdir -p $output_dir

for casename in "${casenames[@]}"; do

    echo "y" | ffmpeg \
        -r 10 \
        -pattern_type glob \
        -i "figures/snapshots_15min_new/lab_sine_forcedry/${casename}/snapshot_?????-?????.png" \
        -c:v libx264       \
        -pix_fmt yuv420p   \
        animation_${casename}.mp4


done
