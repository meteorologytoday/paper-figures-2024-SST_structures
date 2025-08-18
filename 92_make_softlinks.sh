#!/bin/bash

source 00_setup.sh


# Make softlinks to let wnm000-dT000 be the same as wnmXXX-dT000
# This file is run after 91_generate_hourly_avg.sh is finished



for _bl_scheme in MYNN25 MYJ YSU ; do
for target_lab in  lab_SIMPLE lab_FULL ; do 
for U in 10 20 ; do
for wnm in 004 005 007 010 020 040 ; do

    if [[ "$target_lab" =~ "SIMPLE" ]]; then
        mph=off
    elif [[ "$target_lab" =~ "FULL" ]]; then
        mph=on
    fi

    lab_root=$preavg_dir/$target_lab
    actual_casename="case_mph-${mph}_wnm000_U${U}_dT000_${_bl_scheme}"
    softlink_casename="case_mph-${mph}_wnm${wnm}_U${U}_dT000_${_bl_scheme}"

    echo "lab_root = $lab_root"

    pushd .
    cd $lab_root
    if [ -d "$actual_casename" ] ; then

        if [ -e "$softlink_casename" ] ; then
            echo "The softlink $softlink_casename already exists. Remove this..."
            rm -rf $softlink_casename
        fi

        echo "Make softlink $softlink_casename => $actual_casename"
        ln -s ./$actual_casename $softlink_casename

    else

        echo "The actual folder $actual_casename does not exist. Skip this."

    fi
    popd
 
done
done
done
done

