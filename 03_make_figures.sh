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

    # Fig 1
    $sh 11_plot_ocean_SST_analysis.sh                       "BLANK"
    
    # Fig 2
    $sh 12_plot_sounding.sh                                 "BLANK"
 
    # Fig 3
    $sh 15_plot_timeseries_see_steady_state.sh              "BLANK"
   
    # Fig 4 and 5
    $sh 13_plot_system_response.sh                          "BLANK"
    
    # Fig 6
    $sh 19-1_plot_misc_vary_dSST.sh "BLANK"
    $sh 19-2_plot_misc_vary_wnm.sh  "BLANK"

    # Fig 7
    $sh 16-1_plot_dF_flux_decomposition_vary_dSST.sh        "BLANK"
    $sh 16-2_plot_dF_flux_decomposition_vary_wnm.sh         "BLANK"
    
    # Fig 8
    $sh 18-1_plot_spectral_analysis_linearity_on_dSST.sh    "BLANK"
    $sh 18-2_plot_spectral_analysis_wnm1.sh                 "BLANK"

)

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
