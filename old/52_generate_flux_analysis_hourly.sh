#!/bin/bash

source 00_setup.sh

input_dirs=""
exp_names=""
input_dir_root=$data_dir/$target_lab
output_dir=$gendata_dir/flux_decomposition_hourly

dhr=240
time_avg_interval=60   # minutes

mkdir -p $output_dir
N=10

trap "exit" INT TERM
trap "echo 'Exiting... ready to kill jobs... '; kill 0" EXIT

                
for i in 0 ; do
    for Lx in "100" ; do
#    for Lx in "020" "040" "060" "080" "100" "120" "140" "160" "180" "200" ; do
        for U in "20" ; do
#        for U in "00" "05" "10" "15" "20" "25" "30" ; do
            for _bl_scheme in "MYNN25" ; do
#                for dT in 100 ; do
#                for dT in 000 020 040 060 080 100 ; do
                for dT in 000 ; do
                    

                    
                    input_dir="${input_dir_root}/case_mph-off_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}"
                    
                    hrs_beg=$( printf "%02d" $(( $i * $dhr )) )
                    hrs_end=$( printf "%02d" $(( ($i + 1) * $dhr )) )
                    _output_dir="$output_dir/hr${hrs_beg}-${hrs_end}"
                    _output_file="${_output_dir}/case_mph-off_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}.nc"

                    mkdir -p $_output_dir

                    if [ ! -f "$_output_file" ] ; then
                    
                        ((j=j%N)); ((j++==0)) && wait

                        python3 src_dev/gen_flux_analysis_hourly.py  \
                            --input-dir $input_dir  \
                            --output ${_output_file} \
                            --exp-beg-time "2001-01-01 00:00:00" \
                            --time-rng $hrs_beg $hrs_end \
                            --time-avg-interval $time_avg_interval  \
                            --x-rng 0 $Lx          \
                            --wrfout-data-interval 60 \
                            --frames-per-wrfout-file 60 &

                    fi
                    
                done
            done
        done
    done
done


wait

echo "Done"
