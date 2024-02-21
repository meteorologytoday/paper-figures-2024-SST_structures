#!/bin/bash

fig_dir=figures/Cd_analysis

input_dirs=""
exp_names=""
input_dir_root=data/runs


#for dT in 000 010 020 030 040 050 060 070 080 090 100 ; do
for dT in 000 020 040 060 080 100 ; do
    for Lx in "100" ; do
        for U in "15" ; do
            for _bl_scheme in "MYNN25" ; do
                
                input_dirs="$input_dirs ${input_dir_root}/case_mph-off_Lx${Lx}_U${U}_dT${dT}_${_bl_scheme}"
                exp_names="$exp_names $dT"
            done
        done
    done
done

output_dir=$fig_dir
mkdir -p $output_dir
N=2

#dhr=2
#for i in 15 ; do
 
#dhr=6
#for i in 5 ; do
 
#dhr=1
#for i in 30 ; do

ref_exp_order=1 
dmin=360
for i in 5 ; do
    
    ((j=j%N)); ((j++==0)) && wait

    #hrs_beg=$( printf "%02d" $(( $i * $dhr )) )
    #hrs_end=$( printf "%02d" $(( ($i + 1) * $dhr )) )

    mins_beg=$( printf "%02d" $(( $i * $dmin )) )
    mins_end=$( printf "%02d" $(( ($i + 1) * $dmin )) )


    output_name="$output_dir/Cd_analysis_warm_Lnudged_${mins_beg}-${mins_end}.png"
    python3 src/analyze_Cd.py \
        --input-dirs $input_dirs  \
        --exp-beg-time "2001-01-01 00:00:00" \
        --time-rng $mins_beg $mins_end \
        --x-rng 20 30        \
        --exp-names $exp_names \
        --ref-exp-order $ref_exp_order \
        --wrfout-data-interval 60 \
        --frames-per-wrfout-file 60 \
        --output $output_name \
        --time-format hr \
        --extra-title "[warm] " \
        --no-display 


    output_name="$output_dir/Cd_analysis_cold_Lnudged_${mins_beg}-${mins_end}.png"
    python3 src/analyze_Cd.py \
        --input-dirs $input_dirs  \
        --exp-beg-time "2001-01-01 00:00:00" \
        --time-rng $mins_beg $mins_end \
        --x-rng 70 80        \
        --exp-names $exp_names \
        --ref-exp-order $ref_exp_order \
        --wrfout-data-interval 60 \
        --frames-per-wrfout-file 60 \
        --output $output_name \
        --time-format hr \
        --extra-title "[cold] " \
        --no-display 

done

wait

echo "DONE!"
