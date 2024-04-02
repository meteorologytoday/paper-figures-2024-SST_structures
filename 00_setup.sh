#!/bin/bash
py=python3
sh=bash


src_dir=src
data_dir=./data
fig_dir=figures
finalfig_dir=final_figures
simulation_dir=${data_dir}/runs_20240213



echo "src_dir        : $src_dir"
echo "data_dir       : $data_dir"
echo "fig_dir        : $fig_dir"
echo "finalfig_dir   : $finalfig_dir"
echo "simulation_dir : $simulation_dir"



mkdir -p $fig_dir
