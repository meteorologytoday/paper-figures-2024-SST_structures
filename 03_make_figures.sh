#!/bin/bash


nproc=2
while getopts "p:" arg ; do
    case $arg in
        p)
        echo "Set nproc=$OPTARG"
        nproc=$OPTARG
        ;;
    esac
done

echo "nproc = $nproc"

source 98_trapkill.sh
source 00_setup.sh

mkdir -p $fig_dir

plot_codes=(

#    $sh 11_plot_ocean_SST_analysis.sh                   "BLANK"
#    $sh 12_plot_sounding.sh                             "BLANK"
    $sh 13_plot_snapshot.sh                             "BLANK"
)

(


    $sh 15-2_plot_vertical_profile_all_bl_schemes.sh    "BLANK"
    $sh 15-3_plot_vertical_profile_all_bl_schemes_abs   "BLANK"



    $sh 14_plot_timeseries_see_steady_state.sh          "BLANK"
    $sh 14-2_plot_timeseries_relative.sh                "BLANK"
    $sh 16-3_plot_flux_deomposition_vary_dSST.sh        "BLANK"
    $sh 16-4_plot_flux_deomposition_vary_Lx.sh          "BLANK"
    $sh 17_plot_AR_dependency_varying_dSST.sh           "BLANK"
    $sh 17-2_plot_AR_dependency_varying_Lx.sh           "BLANK"
)
#    $sh 13_plot_snapshot.sh "BLANK"
#    $sh 14_plot_spatial_anoamly.sh "BLANK"
#    $sh 15_plot_test_steady_state.sh "BLANK"
#    $sh 16_plot_total_flux_decomposition.sh "BLANK"
#    $sh 18_plot_timeseries-kh.sh "BLANK"
#    $sh 73_plot_WRF_SQ15_phase_fixed_Ug.sh "BLANK"


N=$(( ${#plot_codes[@]} / 3 ))
echo "We have $N file(s) to run..."
for i in $( seq 1 $(( ${#plot_codes[@]} / 3 )) ) ; do
    
    {
        PROG="${plot_codes[$(( (i-1) * 3 + 0 ))]}"
        FILE="${plot_codes[$(( (i-1) * 3 + 1 ))]}"
        OPTS="${plot_codes[$(( (i-1) * 3 + 2 ))]}"
        echo "=====[ Running file: $FILE ]====="
        set -x
        eval "$PROG $FILE $OPTS" 
    } &

    proc_cnt=$(( $proc_cnt + 1))
    
    if (( $proc_cnt >= $nproc )) ; then
        echo "Max proc reached: $nproc"
        wait
        proc_cnt=0
    fi

         
done


wait

echo "Figures generation is complete."
echo "Please run 03_postprocess_figures.sh to postprocess the figures."
