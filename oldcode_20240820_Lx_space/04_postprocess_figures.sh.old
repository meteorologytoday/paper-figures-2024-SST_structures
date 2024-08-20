#!/bin/bash

source 00_setup.sh

finalfig_pdf_dir=$finalfig_dir/pdf
finalfig_png_dir=$finalfig_dir/png



if [ -d "$finalfig_dir" ] ; then
    echo "Output directory '${finalfig_dir}' already exists. Do not make new directory."
else
    echo "Making output directory '${finalfig_dir}'..."
    mkdir $finalfig_dir
fi

mkdir -p $finalfig_pdf_dir
mkdir -p $finalfig_png_dir



echo "Making final figures... "

cp figures_static/* $fig_dir/
python3 postprocess_figures.py --input-dir $fig_dir --output-dir $fig_dir

name_pairs=(
    merged-sst_analysis.png                                               fig01
    merged-experiment_design.png                                          fig02
    test_steady_state/Lx100_U20_dT100_MYNN25/steady_state_test_00-120.png fig03
    timeseries_kh/Lx100_U20_dT100_MYNN25/TOAQOA_timeseries_00-120.png     fig04
    snapshots/Lx100_U20_dT100_MYNN25/snapshot_48-72.png                   fig05
    spatial_anomaly/spatial_anomaly_48-72.png                             fig06
    total_flux_decomposition/dT/decomp_comparison_total_flux_dT_48-72.png fig07
    total_flux_decomposition/Lx/decomp_comparison_total_flux_Lx_48-72.png fig08
    param_space_WRF_SQ15_Ug-20_DTheta-4_RH-0.999_hr48-72.svg              fig09
)


N=$(( ${#name_pairs[@]} / 2 ))
echo "We have $N figure(s) to rename."
for i in $( seq 1 $N ) ; do
    src_file="${name_pairs[$(( (i-1) * 2 + 0 ))]}"
    dst_file="${name_pairs[$(( (i-1) * 2 + 1 ))]}"

    dst_png_file=${dst_file}.png
    dst_pdf_file=${dst_file}.pdf

    echo "### Convert the ${i}-th file:"
    echo "    $src_file => $dst_pdf_file"
    echo "    $src_file => $dst_png_file"

    if [[ "$src_file" =~ \.svg ]]; then
        cairosvg $fig_dir/$src_file -o $finalfig_pdf_dir/$dst_pdf_file
        convert -resize 500 -density 300 $finalfig_pdf_dir/$dst_pdf_file $finalfig_png_dir/$dst_png_file
    elif [[ "$src_file" =~ \.png ]]; then
        convert $fig_dir/$src_file $finalfig_pdf_dir/$dst_pdf_file
        cp $fig_dir/$src_file $finalfig_png_dir/$dst_png_file
    else
        echo "ERROR: Unsupported file extension. File: $src_file"
    fi

done

echo "Done."
