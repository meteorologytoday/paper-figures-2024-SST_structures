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

echo "Figure 3: Merge snapshots... "
svg_stack.py \
    --direction=h \
    $fig_dir/snapshots_dhr-120/lab_sine_DRY/Lx500_U20_dT300_MYNN25/snapshot-part1_120-240.svg \
    $fig_dir/snapshots_dhr-120/lab_sine_WETLWSW/Lx500_U20_dT300_MYNN25/snapshot-part1_120-240.svg \
    > $fig_dir/merged-snapshot_part1.svg

svg_stack.py \
    --direction=v \
    $fig_dir/snapshots_dhr-120/lab_sine_DRY/Lx500_U20_dT300_MYNN25/snapshot-part2_120-240.svg \
    $fig_dir/snapshots_dhr-120/lab_sine_WETLWSW/Lx500_U20_dT300_MYNN25/snapshot-part2_120-240.svg \
    > $fig_dir/merged-snapshot_part2.svg




svg_stack.py \
    --direction=h \
    $fig_dir/snapshots_dhr-120/lab_sine_DRY/Lx100_U20_dT300_MYNN25/snapshot-part1_120-240.svg \
    $fig_dir/snapshots_dhr-120/lab_sine_WETLWSW/Lx100_U20_dT300_MYNN25/snapshot-part1_120-240.svg \
    > $fig_dir/merged-snapshot_100km_part1.svg


svg_stack.py \
    --direction=v \
    $fig_dir/snapshots_dhr-120/lab_sine_DRY/Lx100_U20_dT300_MYNN25/snapshot-part2_120-240.svg \
    $fig_dir/snapshots_dhr-120/lab_sine_WETLWSW/Lx100_U20_dT300_MYNN25/snapshot-part2_120-240.svg \
    > $fig_dir/merged-snapshot_100km_part2.svg

svg_stack.py \
    --direction=v \
    $fig_dir/flux_decomposition_varying_dSST/lab_sine_WETLWSW/flux_decomposition_onefig_varying_dSST_MYNN25_hr120-240.svg  \
    $fig_dir/flux_decomposition_varying_Lx/lab_sine_WETLWSW/flux_decomposition_onefig_varying_Lx_MYNN25_hr120-240.svg \
    > $fig_dir/merged-flux_decomposition_varying_dSST_Lx_MYNN25_hr120-240.svg

svg_stack.py \
    --direction=v \
    $fig_dir/snapshots_vertical_timeseries_WRFV4.6.0_avg-60/VERTICAL_avg-TRUE_Lx500_MYJ_120-240.svg \
    $fig_dir/snapshots_vertical_timeseries_WRFV4.6.0_avg-60/VERTICAL_avg-TRUE_Lx500_YSU_120-240.svg \
    > $fig_dir/merged-snapshots_vertical_timeseres_MYJ-YSU_120-240.svg


svg_stack.py \
    --direction=h \
    $fig_dir/AR_dependency/lab_sine_WETLWSW/AR_dependency_varying_dSST_hr120-240.svg  \
    $fig_dir/AR_dependency/lab_sine_WETLWSW/AR_dependency_varying_Lx_hr120-240.svg    \
    > $fig_dir/merged-AR_dependency_hr120-240.svg


name_pairs=(
    sst_analysis_20170101.svg                                                                     fig01
    merged-exp.svg                                                                                fig02
    merged-snapshot_part1.svg                                                                     fig03
    merged-snapshot_part2.svg                                                                     fig04
    timeseries_WRFV4.6.0/SIMPLE_avg-TRUE_Lx500_MYNN25_timeseries_smooth-25_000-360.svg            fig05
    merged-flux_decomposition_varying_dSST_Lx_MYNN25_hr120-240.svg                                fig06
    merged-AR_dependency_hr120-240.svg                                                            fig07 
    snapshots_vertical_timeseries_WRFV4.6.0_avg-60/VERTICAL_rel_avg-TRUE_Lx100_120-240.svg        fig08
    snapshots_vertical_timeseries_WRFV4.6.0_avg-60/VERTICAL_abs_avg-TRUE_Lx100_120-240.svg        fig09
    merged-snapshot_100km_part1.svg                                                               figS01
    merged-snapshot_100km_part2.svg                                                               figS02
    merged-snapshots_vertical_timeseres_MYJ-YSU_120-240.svg                                       figS03
    timeseries_relative_WRFV4.6.0/HEATFLX_avg-TRUE_Lx500_MYNN25_timeseries_smooth-25_000-360.svg  figS04

#    snapshots_dhr-120/lab_sine_DRY/Lx100_U20_dT300_MYNN25/snapshot-part2_120-240.svg figS02
#    timeseries/AUX_Lx500_timeseries_smooth-25_000-360.svg                            figS03
#    timeseries/AUX_Lx100_timeseries_smooth-25_000-360.svg                            figS04
#    timeseries/HEATFLX_Lx500_timeseries_smooth-25_000-360.svg                        figS05
#    timeseries/HEATFLX_Lx100_timeseries_smooth-25_000-360.svg                        figS06
 
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
