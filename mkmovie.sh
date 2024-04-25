#!/bin/bash


#ffmpeg -framerate 30 -pattern_type glob -i '*.png' \
#  -c:v libx264 -pix_fmt yuv420p out.mp4

#ffmpeg -f concat -i file_list -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p animation_Lx100_U20_dT100_MYNN25.mp4
ffmpeg \
    -r 4 \
    -pattern_type glob \
    -i 'figures/snapshots_15min_new/Lx100_U20_dT100_MYNN25/snapshot_????-????.png' \
    -c:v libx264       \
    -pix_fmt yuv420p   \
    animation_Lx100_U20_dT100_MYNN25.mp4
