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
    $sh 11_plot_ocean_SST_analysis.sh
    
    # Fig 2
    $sh 12_plot_sounding.sh
 
    # Fig 3
    $sh 13_plot_timeseries.sh
   
    # Fig 4
    $sh 14_plot_system_ref.sh

    # Fig 5 and 6
    $sh 15_plot_system_response.sh

    # Fig 7 and 8
    $sh 17_plot_DIV_analysis.sh
    
    # Fig 9
    $sh 19_plot_dF_flux_decomposition_vary_dSST.sh

    # Fig 10
    $sh 20_plot_dF_flux_decomposition_vary_wnm.sh
    
    # Fig 11
    $sh 21_plot_linearity.sh

    # Fig 12
    $sh 22_plot_coherence_analysis_vary_wnm.sh

    # Fig 13
    $sh 23_plot_Ro_analysis.sh
)

nparams=2
N=$(( ${#plot_codes[@]} / $nparams ))
echo "We have $N file(s) to run..."
for i in $( seq 1 $(( ${#plot_codes[@]} / $nparams )) ) ; do
   
    echo "#### This is the $i-th command. ####"
 
    {
        PROG="${plot_codes[$(( (i-1) * $nparams + 0 ))]}"
        FILE="${plot_codes[$(( (i-1) * $nparams + 1 ))]}"
        echo "=====[ Running file: $FILE ]====="
        cmd="$PROG $FILE" 
        echo ">> $cmd"
        eval "$cmd"
        echo "Return code $?"
    } &

    proc_cnt=$(( $proc_cnt + 1 ))
    
    if (( $proc_cnt >= $nproc )) ; then
        echo "Max proc reached: $nproc"
        wait
        proc_cnt=0
    fi

         
done


wait

echo "Figures generation is complete."
echo "Please run 04_postprocess_figures.sh to postprocess the figures."
