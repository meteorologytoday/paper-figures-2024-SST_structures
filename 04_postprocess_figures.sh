#!/bin/bash

source 00_setup.sh


finalfig_pdf_dir=$finalfig_dir/pdf
finalfig_png_dir=$finalfig_dir/png


echo "Making output directory '${finalfig_dir}'..."
mkdir -p $finalfig_dir
mkdir -p $finalfig_pdf_dir
mkdir -p $finalfig_png_dir


echo "Making final figures... "

#echo "Figure 1: Merge SST map and spectrum analysis... "
#svg_stack.py --direction=h $fig_dir/sst_analysis_map_20170101.svg $fig_dir/sst_analysis_spec_20170101.svg > $fig_dir/merged-sst_analysis.svg


name_pairs=(
    sst_analysis_20170101.svg                fig01
    input_sounding_woML.svg                  fig02
)

N=$(( ${#name_pairs[@]} / 2 ))
echo "We have $N figure(s) to rename and convert into pdf files."
for i in $( seq 1 $N ) ; do

    {
    src_file="${name_pairs[$(( (i-1) * 2 + 0 ))]}"
    dst_file_pdf="${name_pairs[$(( (i-1) * 2 + 1 ))]}.pdf"
    dst_file_png="${name_pairs[$(( (i-1) * 2 + 1 ))]}.png"
    
    echo "$src_file => $dst_file_pdf"
    cairosvg $fig_dir/$src_file -o $finalfig_pdf_dir/$dst_file_pdf

    echo "$src_file => $dst_file_png"
    convert $finalfig_pdf_dir/$dst_file_pdf $finalfig_png_dir/$dst_file_png
    } &
done

wait

echo "Done."
