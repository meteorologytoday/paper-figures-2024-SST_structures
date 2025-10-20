#!/bin/bash
py=python3
sh=bash


src_dir=src
data_dir=./data
data_sim_dir=$data_dir/sim_data
fig_dir=figures
fig_static_dir=figures_static
finalfig_dir=./final_figures


target_lab=lab_sine_forcedry
bl_schemes=( MYNN25 )

fig_ext=svg

Us=(
    20
    10
)


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

function gen_gendata_dir {
    echo "./gendata/gendata_U$1"
}

function gen_preavg_dir {
    echo $( gen_gendata_dir $1 )/preavg
}

function gen_delta_analysis_dir {
    echo $( gen_gendata_dir $1 )/delta_analysis_style-$2
}

