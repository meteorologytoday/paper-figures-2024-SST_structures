#!/bin/bash

source 00_setup.sh

for U in "${Us[@]}" ; do

echo " ===== Plotting U = ${U} m/s ===== "

finalfig_dir=$( gen_final_figures_dir $U )
finalfig_pdf_dir=$finalfig_dir/pdf
finalfig_png_dir=$finalfig_dir/png
finalfig_svg_dir=$finalfig_dir/svg


echo "Making output directory '${finalfig_dir}'..."
mkdir -p $finalfig_dir
mkdir -p $finalfig_pdf_dir
mkdir -p $finalfig_png_dir
mkdir -p $finalfig_svg_dir


echo "Making final figures... "

echo "Figure 2: Merge experiment design and vertical profile..."
svg_stack.py                                \
    --direction=h                           \
    $fig_static_dir/experiment_design.svg \
    $fig_dir/sounding_U${U}.svg        \
    > $fig_dir/merged-exp.svg



echo "Figure 4: Merge vertical... "
for bl_scheme in MYNN25 MYJ YSU ; do 
for dT in 000; do
for wnm in 000 ; do
   
    hrs_beg=240
    dhr=$( get_dhr $bl_scheme ) 
    hrs_end=$(( $hrs_beg + $dhr ))
    hrs=${hrs_beg}-${hrs_end}
 
    svg_stack.py \
        --direction=v \
        $fig_dir/snapshots-full_dhr-$dhr/lab_SIMPLE/case_mph-off_wnm${wnm}_U${U}_dT${dT}_${bl_scheme}/snapshot-full-vertical-profile_${hrs}.svg \
        $fig_dir/snapshots-full_dhr-$dhr/lab_FULL/case_mph-on_wnm${wnm}_U${U}_dT${dT}_${bl_scheme}/snapshot-full-vertical-profile_${hrs}.svg \
        > $fig_dir/merged-snapshot-vertical-profile_wnm${wnm}_U${U}_dT${dT}_${bl_scheme}.svg
done
done
done



echo "Figure 5 and 6: Merge snapshots... "
for bl_scheme in MYNN25 MYJ YSU ; do 
for dT in 100; do
for wnm in 010 ; do
 
    hrs_beg=240
    dhr=$( get_dhr $bl_scheme ) 
    hrs_end=$(( $hrs_beg + $dhr ))
    hrs=${hrs_beg}-${hrs_end}
    
    svg_stack.py \
        --direction=h \
        $fig_dir/snapshots_dhr-${dhr}/lab_SIMPLE/case_mph-off_wnm${wnm}_U${U}_dT${dT}_${bl_scheme}/snapshot-part1_${hrs}.svg \
        $fig_dir/snapshots_dhr-${dhr}/lab_FULL/case_mph-on_wnm${wnm}_U${U}_dT${dT}_${bl_scheme}/snapshot-part1_${hrs}.svg \
        > $fig_dir/merged-snapshot_wnm${wnm}_U${U}_dT${dT}_${bl_scheme}_part1.svg

    svg_stack.py \
        --direction=v \
        $fig_dir/snapshots_dhr-${dhr}/lab_SIMPLE/case_mph-off_wnm${wnm}_U${U}_dT${dT}_${bl_scheme}/snapshot-part2_${hrs}.svg \
        $fig_dir/snapshots_dhr-${dhr}/lab_FULL/case_mph-on_wnm${wnm}_U${U}_dT${dT}_${bl_scheme}/snapshot-part2_${hrs}.svg \
        > $fig_dir/merged-snapshot_wnm${wnm}_U${U}_dT${dT}_${bl_scheme}_part2.svg

done
done
done

echo "Merge divergence tendency analysis... "
for bl_scheme in MYNN25 MYJ YSU ; do 
for dT in 100; do
for target_lab in SIMPLE FULL; do

    hrs_beg=240
    dhr=$( get_dhr $bl_scheme ) 
    hrs_end=$(( $hrs_beg + $dhr ))
    hrs=${hrs_beg}-${hrs_end}
 
    if [ "$target_lab" = "FULL" ] ; then
        mph="on"
    else
        mph="off"
    fi
    
    svg_stack.py \
        --direction=h \
        $fig_dir/div_analysis_dhr-${dhr}/lab_${target_lab}/case_mph-${mph}_wnm010_U${U}_dT100_${bl_scheme}/div_analysis_${hrs}_halfwindow-10.svg \
        $fig_dir/div_analysis_dhr-${dhr}/lab_${target_lab}/case_mph-${mph}_wnm005_U${U}_dT100_${bl_scheme}/div_analysis_${hrs}_halfwindow-10.svg \
        > $fig_dir/merged-div_analysis_lab_${target_lab}_U${U}_dT${dT}_${bl_scheme}.svg
done
done
done


echo "Merge linearity"
for bl_scheme in MYNN25 MYJ YSU ; do 
    hrs_beg=240
    dhr=$( get_dhr $bl_scheme ) 
    hrs_end=$(( $hrs_beg + $dhr ))
    hrs=${hrs_beg}-${hrs_end}
 
    svg_stack.py      \
       --direction=h \
        $fig_dir/linearity_analysis/linearity_vary_wnm_lab_SIMPLE_dSST100_U${U}_${bl_scheme}_hr${hrs}.svg \
        $fig_dir/linearity_analysis/linearity_vary_wnm_lab_FULL_dSST100_U${U}_${bl_scheme}_hr${hrs}.svg \
        > $fig_dir/merged-linearity_vary_wnm_${bl_scheme}_U${U}_hr${hrs}.svg
done

name_pairs=(
    sst_analysis_20240101.svg                                                                        fig01
    merged-exp.svg                                                                                   fig02
    timeseries/timeseries_wnm010_U${U}_dT100_MYNN25_timeseries_smooth-25_000-360.svg                 fig03
    merged-snapshot-vertical-profile_wnm000_U${U}_dT000_MYNN25.svg                                   fig04
    merged-snapshot_wnm010_U${U}_dT100_MYNN25_part1.svg                                              fig05
    merged-snapshot_wnm010_U${U}_dT100_MYNN25_part2.svg                                              fig06
    merged-div_analysis_lab_SIMPLE_U${U}_dT100_MYNN25.svg                                            fig07 
    merged-div_analysis_lab_FULL_U${U}_dT100_MYNN25.svg                                              fig08 
    dF_flux_decomposition_varying_dSST/lab_FULL/dF_flux_decomposition_onefig_U${U}_wnm010_varying_dSST_MYNN25_hr240-360.svg  fig09
    dF_flux_decomposition_varying_wnm/lab_FULL/dF_flux_decomposition_onefig_U${U}_dSST100_varying_wnm_MYNN25_hr240-360.svg   fig10

    merged-linearity_vary_wnm_MYNN25_U${U}_hr240-360.svg                                             fig11
    coherence_analysis/coherence_on_dSST_vary_wnm_U${U}_dSST100_MYNN25_hr240-360.svg                 fig12
                       coherence_on_dSST_vary_wnm_U20_dSST100_MYNN25_hr240-360.svg0
    Ro_analysis/Ro_analysis_U${U}_vary_wnm_dSST100_MYNN25_hr240-360.svg                              fig13

    timeseries/timeseries_wnm010_U20_dT100_MYJ_timeseries_smooth-25_000-360.svg                      figS01
    merged-snapshot-vertical-profile_wnm000_U20_dT000_MYJ.svg                                        figS02
    merged-snapshot_wnm010_U20_dT100_MYJ_part1.svg                                                   figS03
    merged-snapshot_wnm010_U20_dT100_MYJ_part2.svg                                                   figS04
    dF_flux_decomposition_varying_dSST/lab_FULL/dF_flux_decomposition_onefig_wnm010_varying_dSST_MYJ_hr120-168.svg  figS05
    dF_flux_decomposition_varying_wnm/lab_FULL/dF_flux_decomposition_onefig_dSST100_varying_wnm_MYJ_hr120-168.svg   figS06

    #merged-div_analysis_lab_SIMPLE_dT100_MYJ.svg                                                     figS05 
    #merged-div_analysis_lab_FULL_dT100_MYJ.svg                                                       figS06 
    #merged-linearity_vary_wnm_MYJ_hr120-168.svg                                                      figS09
    #coherence_analysis/coherence_on_dSST_vary_wnm_dSST100_MYJ_hr120-168.svg                          figS10
    #Ro_analysis/Ro_analysis_vary_wnm_dSST100_MYJ_hr120-168.svg                                       figS11


    timeseries/timeseries_wnm010_U20_dT100_YSU_timeseries_smooth-25_000-360.svg                      figS07
    merged-snapshot-vertical-profile_wnm000_U20_dT000_YSU.svg                                        figS08
    merged-snapshot_wnm010_U20_dT100_YSU_part1.svg                                                   figS09
    merged-snapshot_wnm010_U20_dT100_YSU_part2.svg                                                   figS10
    dF_flux_decomposition_varying_dSST/lab_FULL/dF_flux_decomposition_onefig_wnm010_varying_dSST_YSU_hr240-360.svg  figS11
    dF_flux_decomposition_varying_wnm/lab_FULL/dF_flux_decomposition_onefig_dSST100_varying_wnm_YSU_hr240-360.svg   figS12

    #merged-div_analysis_lab_SIMPLE_dT100_YSU.svg                                                     figS16 
    #merged-div_analysis_lab_FULL_dT100_YSU.svg                                                       figS17 
    #merged-linearity_vary_wnm_YSU_hr240-360.svg                                                      figS20
    #coherence_analysis/coherence_on_dSST_vary_wnm_dSST100_YSU_hr240-360.svg                          figS21
    #Ro_analysis/Ro_analysis_vary_wnm_dSST100_YSU_hr240-360.svg                                       figS22

#    merged-phase_misc_hr240-360.svg                                                                  fig07
#    DIV_analysis_tracking_wnm/DIV_analysis_lab_FULL_dT100_MYNN25_hr240-360.svg                       fig08

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

done

echo "Done."
