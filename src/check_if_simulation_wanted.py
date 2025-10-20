import xarray as xr
import pandas as pd
import numpy as np
import argparse
import wrf_load_helper 
from pathlib import Path

import datetime
import sys

def check_if_simulation_wanted(
    target_lab,
    wnm,
    dT,
    U,
):

    result = True
    
    if U in [15, 20]:
        if target_lab in ["FULL", "lab_FULL"]:

            if wnm == 0 and dT == 0:
                pass
            elif (dT == 0) != (wnm == 0):
                result = False
            elif not ( wnm in [10, ] or dT in [1,] ):
                result = False

        elif target_lab in ["SIMPLE", "lab_SIMPLE"]:

            if wnm == 0 and dT == 0:
                pass
            elif (dT == 0) != (wnm == 0):
                result = False
            elif (wnm, dT) not in [ (5, 1.0) , (10, 1.0) ]:
                result = False
 
    elif U in [10,]:  # only needed for DIV analysis

        if wnm == 0 and dT == 0:
            pass
        elif not ( wnm in [5, 10,] and dT == 1.0 ):
            result = False
    else:
        result = False
        
    return result 



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--target-lab',   type=str,   help='Target lab', required=True)
    parser.add_argument('--dT100',        type=int,   help='dSST * 100', required=True)
    parser.add_argument('--wnm',          type=int,   help='Wavenumber', required=True)
    parser.add_argument('--U',            type=int,   help='Backaground wind', required=True)
    args = parser.parse_args()

    result = check_if_simulation_wanted(
        target_lab = args.target_lab,
        wnm = args.wnm,
        dT = args.dT100 / 100.0,
        U  = args.U,
    )

    if result == True:
        print("TRUE")
    else:
        print("FALSE")


