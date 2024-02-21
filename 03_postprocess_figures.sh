#!/bin/bash


source 00_setup.sh



if [ -d "$finalfig_dir" ] ; then
    echo "Output directory '${finalfig_dir}' already exists. Do not make new directory."
else
    echo "Making output directory '${finalfig_dir}'..."
    mkdir $finalfig_dir
fi



echo "Making final figures... "

convert $fig_dir/sst_analysis_map.png \
        $fig_dir/sst_analysis_spec.png \
        -gravity Northwest +append $fig_dir/merged-sst_analysis.png

name_pairs=(
    merged-sst_analysis.png                                               fig01.png
    input_sounding_woML.png                                               fig02.png
    snapshots/Lx100_U15_dT100_MYNN25/snapshot_24-30.png                   fig03.png
    test_steady_state/Lx100_U15_dT100_MYNN25/steady_state_test_00-72.png  fig04.png
    spatial_anomaly/spatial_anomaly_24-30.png                             fig05.png
    decomp_comparison_total_flux_24-30.png                                fig06.png
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
