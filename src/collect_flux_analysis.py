import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import wrf_load_helper 
import datetime
import os

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-dir', type=str, help='Input directories.', required=True)
    parser.add_argument('--output', type=str, help='Output filename in png.', default="")
    parser.add_argument('--file-fmt', type=str, help='File format using pythons string format.', required=True)

    parser.add_argument('--Ugs', type=int, nargs="+", help='Ug in m/s.', required=True)
    parser.add_argument('--Lxs', type=int, nargs="+", help='Lx in km.', required=True)
    parser.add_argument('--dSSTs', type=float, nargs="+", help='dSST in K.', required=True)

    args = parser.parse_args()

    print(args)


    varnames = [
        "LH", "HFX",
        "WND_TOA_cx", "WND_QOA_cx",
    ]


    data = xr.Dataset(
        data_vars = {
            k : ( ["Ug", "Lx", "dSST"], np.zeros((len(args.Ugs), len(args.Lxs), len(args.dSSTs))) )
            for k in varnames
        },

        coords = dict(
            Ug = (["Ug"], args.Ugs),
            Lx = (["Lx"], args.Lxs),
            dSST = (["dSST"], args.dSSTs),
        )
    )
    
    
    for i, Ug in enumerate(args.Ugs):
        for j, Lx in enumerate(args.Lxs):
            for k, dSST in enumerate(args.dSSTs):
                
                filename = os.path.join(
                    args.input_dir,
                    args.file_fmt.format(
                        Ug = Ug,
                        Lx = Lx,
                        dSST = int(dSST),
                    )
                )

                print("Loading file: ", filename)
                _ds = xr.open_dataset(filename)
                for varname in varnames:
                    data[varname][i, j, k] = _ds[varname]

    
    print("Output file: ", args.output)
    data.to_netcdf(args.output)

                
