#!/bin/bash

source 00_setup.sh

input_dirs=""
exp_names=""
parameter=dT
input_dir_root=$gendata_dir/flux_decomposition_hourly
output_dir=$fig_dir/flux_decomposition_timeseries/vary_$parameter

#trap "exit" INT TERM
#trap "echo 'Exiting... ready to kill jobs... '; kill 0" EXIT



mkdir -p $output_dir
N=2

dhr=6
#for i in $( seq 0 0 ); do
for i in 4 ; do
    
    ((j=j%N)); ((j++==0)) && wait

    hrs_beg=$( printf "%02d" $(( $i * $dhr )) )
    hrs_end=$( printf "%02d" $(( ($i + 1) * $dhr )) )

    output_name="$output_dir/decomp_timeseries_${parameter}_${hrs_beg}-${hrs_end}.png"

    #for dT in 000 010 020 030 040 050 060 070 080 090 100 ; do
    #for dT in 000 020 040 060 080 100 ; do
    for dT in 000 020 040 060 080 100 ; do
        for Lx in "100" ; do
            for U in "20" ; do
                for _bl_scheme in "MYNN25" ; do
                    
                    input_files="$input_files ${input_dir_root}/hr${hrs_beg}-${hrs_end}/case_mph-off_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}.nc"
                    exp_names="$exp_names $dT"
                done
            done
        done
    done


    python3 src/plot_flux_decomposition_timeseries.py  \
        --input-files $input_files  \
        --exp-beg-time "2001-01-01 00:00:00" \
        --HFX-rng -15 10        \
        --LH-rng 10 35        \
        --exp-names $exp_names \
        --output $output_name \
        --no-display 


done

wait

echo "DONE!"
