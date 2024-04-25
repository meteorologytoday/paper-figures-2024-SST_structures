#!/bin/bash

source 00_setup.sh

if [ -d "$finalfig_dir" ] ; then
    echo "Output directory '${finalfig_dir}' already exists. Do not make new directory."
else
    echo "Making output directory '${finalfig_dir}'..."
    mkdir $finalfig_dir
fi



echo "Making final figures... "

cp figures_static/* $fig_dir/
python3 postprocess_figures.py --input-dir $fig_dir --output-dir $fig_dir

name_pairs=(
    merged-sst_analysis.png                                               fig01.png
    merged-experiment_design.png                                          fig02.png
    test_steady_state/Lx100_U20_dT100_MYNN25/steady_state_test_00-120.png fig03.png
    timeseries_kh/Lx100_U20_dT100_MYNN25/TOAQOA_timeseries_00-120.png     fig04.png
    snapshots/Lx100_U20_dT100_MYNN25/snapshot_48-72.png                   fig05.png
    spatial_anomaly/spatial_anomaly_48-72.png                             fig06.png
    total_flux_decomposition/dT/decomp_comparison_total_flux_dT_48-72.png fig07.png
    total_flux_decomposition/Lx/decomp_comparison_total_flux_Lx_48-72.png fig08.png
)


N=$(( ${#name_pairs[@]} / 2 ))
echo "We have $N figure(s) to rename."
for i in $( seq 1 $N ) ; do
    src_file="${name_pairs[$(( (i-1) * 2 + 0 ))]}"
    dst_file="${name_pairs[$(( (i-1) * 2 + 1 ))]}"
    echo "$src_file => $dst_file"
    cp $fig_dir/$src_file $finalfig_dir/$dst_file 
done

echo "Done."
