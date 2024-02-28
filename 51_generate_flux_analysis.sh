#!/bin/bash

source 00_setup.sh

input_dirs=""
exp_names=""
input_dir_root=$data_dir/$target_lab
output_dir=$gendata_dir/flux_decomposition

dhr=24

mkdir -p $output_dir
N=1

trap "exit" INT TERM
trap "echo '!!!!'; kill 0" EXIT


for Lx in "050" "100" "200" ; do
    for U in "15" ; do
        for _bl_scheme in "MYNN25" ; do
            input_dirs=""
            for dT in 000 020 040 060 080 100 ; do
                input_dirs="$input_dirs ${input_dir_root}/case_mph-off_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}"
                exp_names="$exp_names $dT"
            done
        done


        for i in 1 2 0; do
            echo "Hi, $i"
            
            ((j=j%N)); ((j++==0)) && wait

            hrs_beg=$( printf "%02d" $(( $i * $dhr )) )
            hrs_end=$( printf "%02d" $(( ($i + 1) * $dhr )) )
            _output_dir="$output_dir/expset_hr${hrs_beg}-${hrs_end}_mph-off_Lx${Lx}_U${U}_${_bl_scheme}"
            mkdir -p $_output_dir

            python3 src/gen_flux_analysis.py  \
                --input-dirs $input_dirs  \
                --exp-beg-time "2001-01-01 00:00:00" \
                --time-rng $hrs_beg $hrs_end \
                --x-rng 0 $Lx          \
                --ref-exp-order 0 \
                --wrfout-data-interval 60 \
                --frames-per-wrfout-file 60 \
                --output-dir $_output_dir &


        done

    done
done


wait


echo "DONE!"

for decomp in HFX LH ; do
    for U in "15" ; do
        for _bl_scheme in "MYNN25" ; do
        
            for i in 1 2 0; do

                hrs_beg=$( printf "%02d" $(( $i * $dhr )) )
                hrs_end=$( printf "%02d" $(( ($i + 1) * $dhr )) )

                input_files=""
                exp_names=""
                for Lx in "050" "100" "200" ; do
                    input_dir="$output_dir/expset_hr${hrs_beg}-${hrs_end}_mph-off_Lx${Lx}_U${U}_${_bl_scheme}"
                    input_files="$input_files ${input_dir}/diff_analysis_${decomp}.nc"
                    exp_names="$exp_names $Lx"
                done

                _output_file="$fig_dir/${decomp}_expset_hr${hrs_beg}-${hrs_end}_mph-off_U${U}_${_bl_scheme}.png"

                python3 src/plot_comparison_decomposition.py  \
                    --input-files $input_files  \
                    --labels $exp_names         \
                    --decomp-type $decomp       \
                    --no-display \
                    --output $_output_file

            done
        done
    done
done

