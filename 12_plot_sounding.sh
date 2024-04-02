#!/bin/bash

source 00_setup.sh

SST_base=288.15 # 15 degC
RH=90
U=15
deltaGamma=3e-3
z_st=10000.0

python3 $src_dir/make_ideal_sounding_simple.py \
    --output $data_dir/input_sounding_woML.ref \
    --Delta-Gamma $deltaGamma \
    --U $U   \
    --RH $RH \
    --T-sfc $SST_base  \
    --z-st  $z_st \
    --output-fig $fig_dir/input_sounding_woML.png \
    --thumbnail-numbering bcdef \
    --no-display

