#!/bin/bash

source 00_setup.sh


finalfig_pdf_dir=$finalfig_dir/pdf
finalfig_png_dir=$finalfig_dir/png
finalfig_svg_dir=$finalfig_dir/svg


echo "Making output directory '${finalfig_dir}'..."
mkdir -p $finalfig_dir
mkdir -p $finalfig_pdf_dir
mkdir -p $finalfig_png_dir
mkdir -p $finalfig_svg_dir


echo "Making final figures... "

#echo "Figure 1: Merge SST map and spectrum analysis... "
#svg_stack.py --direction=h $fig_dir/sst_analysis_map_20170101.svg $fig_dir/sst_analysis_spec_20170101.svg > $fig_dir/merged-sst_analysis.svg


echo "Figure 2: Merge experiment design and vertical profile..."
svg_stack.py \
    --direction=h \
    $fig_static_dir/experiment_design.svg \
    $fig_dir/input_sounding_woML.svg \
    > $fig_dir/merged-exp.svg

name_pairs=(
    sst_analysis_20170101.svg                                                        fig01
    merged-exp.svg                                                                   fig02
    snapshots_dhr-120/lab_sine_dry/Lx500_U20_dT300_MYNN25/snapshot-part1_120-240.svg fig03
    snapshots_dhr-120/lab_sine_dry/Lx500_U20_dT300_MYNN25/snapshot-part2_120-240.svg fig04
    timeseries/SIMPLE_Lx500_timeseries_smooth-25_000-360.svg                         fig05
    timeseries_relative/HEATFLX_Lx500_timeseries_smooth-25_000-360.svg               fig06
    flux_decomposition_varying_dSST/lab_sine_wetlwsw/flux_decomposition_varying_dSST_MYNN25_hr240-360.svg fig07
    flux_decomposition_varying_Lx/lab_sine_wetlwsw/flux_decomposition_varying_Lx_MYNN25_hr240-360.svg     fig08
    snapshots_dhr-120/lab_sine_dry/Lx100_U20_dT300_MYNN25/snapshot-part1_120-240.svg figS01
    snapshots_dhr-120/lab_sine_dry/Lx100_U20_dT300_MYNN25/snapshot-part2_120-240.svg figS02
    timeseries/AUX_Lx500_timeseries_smooth-25_000-360.svg                            figS03
    timeseries/AUX_Lx100_timeseries_smooth-25_000-360.svg                            figS04
    timeseries/HEATFLX_Lx500_timeseries_smooth-25_000-360.svg                        figS05
    timeseries/HEATFLX_Lx100_timeseries_smooth-25_000-360.svg                        figS06
 
)

N=$(( ${#name_pairs[@]} / 2 ))
echo "We have $N figure(s) to rename and convert into pdf files."
for i in $( seq 1 $N ) ; do

    {

    src_file="${name_pairs[$(( (i-1) * 2 + 0 ))]}"
    dst_file_pdf="${name_pairs[$(( (i-1) * 2 + 1 ))]}.pdf"
    dst_file_png="${name_pairs[$(( (i-1) * 2 + 1 ))]}.png"
    dst_file_svg="${name_pairs[$(( (i-1) * 2 + 1 ))]}.svg"
 
    echo "$src_file => $dst_file_svg"
    cp $fig_dir/$src_file $finalfig_svg_dir/$dst_file_svg
   
    echo "$src_file => $dst_file_pdf"
    cairosvg $fig_dir/$src_file -o $finalfig_pdf_dir/$dst_file_pdf

    echo "$src_file => $dst_file_png"
    magick $finalfig_pdf_dir/$dst_file_pdf $finalfig_png_dir/$dst_file_png

    } &
done

wait

echo "Done."
