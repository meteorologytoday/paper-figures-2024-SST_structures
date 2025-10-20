#!/bin/bash

source 00_setup.sh

wnms="004 005 007 010 020 040"

dSSTs="000 010 030 050 100 150 200 250 300"
Us=( 20 )
bl_schemes=(
    MYNN25
#    YSU
#    MYJ
)

target_labs=(
    lab_SIMPLE
    lab_FULL
)

input_params=(
#    wnm  010   $(( 24 * 5 )) $(( 24 * 10 ))
#    dSST 100   $(( 24 * 5 )) $(( 24 * 10 ))
#    dSST 050   $(( 24 * 5 )) $(( 24 * 10 ))

    wnm  010   $(( 24 * 10 )) $(( 24 * 15 ))
    dSST 100   $(( 24 * 10 )) $(( 24 * 15 ))
)


nparams=4
nproc=1




N=$(( ${#input_params[@]} / $nparams ))
for i in $( seq 1 $N ) ; do
    
    fixed_param="${input_params[$(( (i-1) * $nparams + 0 ))]}"
    fixed_value="${input_params[$(( (i-1) * $nparams + 1 ))]}"
    hrs_beg="${input_params[$(( (i-1) * $nparams + 2 ))]}"
    hrs_end="${input_params[$(( (i-1) * $nparams + 3 ))]}"

    echo "fixed_param = $fixed_param"
    echo "fixed_value = $fixed_value"
    echo "hrs_beg = $hrs_beg"
    echo "hrs_end = $hrs_end"


    if [[ "$fixed_param" =~ "wnm" ]] ; then
        
        _wnms=$fixed_value
        _dSSTs=$dSSTs

    elif [[ "$fixed_param" =~ "dSST" ]] ; then
        
        _wnms=$wnms
        _dSSTs=$fixed_value

    else 
    
        echo "ERROR: unknown fixed_param : $fixed_param"
        exit 1
    fi

    for Ug in "${Us[@]}"; do
    for bl_scheme in "${bl_schemes[@]}" ; do
    for target_lab in "${target_labs[@]}" ; do       
        
        if [[ "$target_lab" =~ "FULL" ]]; then
            mph=on
        elif [[ "$target_lab" =~ "SIMPLE" ]]; then
            mph=off
        fi

        gendata_dir=$( gen_gendata_dir $Ug )
        analysis_root=$( gen_delta_analysis_dir $Ug STYLE1 )
        output_root=$gendata_dir/dF_phase_analysis/fixed_${fixed_param}

        casename=case_mph-${mph}_wnm${wnm}_U${U}_dT${dT}_${bl_scheme}

        input_dir_fmt=$analysis_root/$target_lab/case_mph-${mph}_wnm{wnm:s}_U{Ug:s}_dT{dSST:s}_${bl_scheme}/avg_before_analysis-TRUE

        output_dir=$output_root/${target_lab}
        output_file=$output_dir/collected_flux_U${Ug}_${bl_scheme}_hr${hrs_beg}-${hrs_end}.nc

        mkdir -p $output_dir

        if [ -f "$output_file" ] ; then
            echo "File $output_file exists! Skip."
            continue
        fi

        eval "python3 src/collect_flux_analysis_wnm.py  \
            --input-dir-fmt $input_dir_fmt  \
            --output $output_file \
            --Ugs $Ug \
            --wnms $_wnms \
            --dSSTs $_dSSTs \
            --time-rng $hrs_beg $hrs_end \
            --moving-avg-cnt 25 \
            --expand-time-min $(( 24 * 60 )) \
            --exp-beg-time 2001-01-01T00:00:00 \
            --wrfout-data-interval 3600 \
            --frames-per-wrfout-file 12 \
        " &

        proc_cnt=$(( $proc_cnt + 1))
        
        if (( $proc_cnt >= $nproc )) ; then
            echo "Max proc reached: $nproc"
            wait
            proc_cnt=0
        fi


    done
    done
    done
done   
 
echo "Done"
