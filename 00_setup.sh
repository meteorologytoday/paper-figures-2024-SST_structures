#!/bin/bash
py=python3
sh=bash


src_dir=src
data_dir=./data
data_sim_dir=$data_dir/sim_data
data_SQ15_dir=./data/data_SQ15
fig_dir=figures
fig_static_dir=figures_static
finalfig_dir=final_figures


target_lab=lab_sine_forcedry
bl_schemes=( MYNN25 )

gendata_dir=./gendata
preavg_dir=$gendata_dir/preavg

fig_ext=svg

Lx=2000
f0=1e-4
mkdir -p $fig_dir


function get_dhr {
    
    if [ "$1" = "MYJ" ] ; then
        echo $(( 2 * 24 ))
    else
        echo $(( 5 * 24 ))
    fi
}
