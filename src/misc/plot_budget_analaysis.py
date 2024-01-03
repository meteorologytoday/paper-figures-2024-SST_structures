import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import diagnostics
import wrf_load_helper 
import datetime
import traceback
from multiprocessing import Pool

import os
from pathlib import Path


wrf_load_helper.engine = "netcdf4"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--title', type=str, help='analysis beg time', default="")
parser.add_argument('--title-template', type=str, help='analysis beg time', default="time")
parser.add_argument('--input-dir', type=str, help='Input directories.', required=True)
parser.add_argument('--output-dir', type=str, help='Output filename in png.', required=True)
parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
parser.add_argument('--z-idx', type=int, help='Z-idx to be plotted.', required=True)
parser.add_argument('--nproc', type=int, help='Number of processes.', default=1)
parser.add_argument('--time-rng', type=int, nargs=2, help="Time range to process in minutes after --exp-beg-time", required=True)
parser.add_argument('--overlay', action="store_true")
parser.add_argument('--no-display', action="store_true")

args = parser.parse_args()

print(args)

exp_beg_time = pd.Timestamp(args.exp_beg_time)

time_beg = exp_beg_time + pd.Timedelta(minutes=args.time_rng[0])
time_end = exp_beg_time + pd.Timedelta(minutes=args.time_rng[1])
wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)

wsm = wrf_load_helper.WRFSimMetadata(
    start_datetime  = exp_beg_time,
    data_interval   = wrfout_data_interval,
    frames_per_file = args.frames_per_wrfout_file,
)

# =================================================================

print("Loading matplotlib...")
import matplotlib
if args.no_display:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
print("Done")

def plotBudget(dt_beg, dt_end, output_filename):

    result = dict(status="UNKNOWN", dt_beg=dt_beg, dt_end=dt_end, output_filename=output_filename)

    try:

        print("Start loading data.")
       
        ds = wrf_load_helper.loadWRFDataFromDir(
            wsm, 
            args.input_dir,
            beg_time = dt_beg,
            end_time = dt_end,
            prefix="wrf_budget_analysis_",
            suffix=".nc",
            avg=False,
            verbose=False,
            inclusive="both",
        )

        print("Done loading data.")        
            
        ds_time_beg = pd.Timestamp(ds.time.to_numpy()[0]).strftime("%d %H:%M:%S")
        ds_time_end = pd.Timestamp(ds.time.to_numpy()[-1]).strftime("%d %H:%M:%S")
        ds = ds.mean(dim=['time'], keep_attrs=True)

        ds = ds.isel(z=args.z_idx)

        varnames = [
            ("dUdt_T",      "+"),
            ("UdUdX_T",     "-"),
            ("WdUdZ_T",     "-"),
            ("fV_T",        "+"), 
            ("dPdX_T",      "-"),
            ("RNLDSTRS_T",  "+"),
            ("Umom_res",    "+"),
        ]

        fig, ax = plt.subplots( ( 1 if args.overlay else len(varnames) ), 1, figsize=(10, 6), sharex=True, squeeze=False)
        x = ds.coords['X'] / 1e3

        title = args.title

        if args.title_template == "time":
            title = "Date: %s ~ %s. ( $ \\eta = %.3f$ )" % (ds_time_beg, ds_time_end, ds.ETA.to_numpy().item())

        fig.suptitle(title)

        for i, (varname, sign) in enumerate(varnames):
            
            if args.overlay:
                _ax = ax[0, 0]
            else:
                _ax = ax[i, 0]
                _ax.set_title(varname)

            _sign = None
            if sign == "+":
                _sign = 1.0
                _sign_str = ""
            elif sign == "-":
                _sign = -1.0
                _sign_str = "- "

            _ax.plot(x, _sign * ds[varname], label="%s%s" % (_sign_str, varname))

            _ax.grid()
            _ax.set_ylim(np.array([-1, 1]) * 1e-3)
            _ax.set_ylabel("[ $\\mathrm{m} / \\mathrm{s}^2 $ ]")


        if args.overlay:
            ax[0, 0].legend(loc="lower left")
            
        ax[-1, 0].set_xlabel("[ $\\mathrm{km}$ ]")

        ax[0, 0].set_xlim([0, 1000])

        print("Saving output: ", output_filename)
        fig.savefig(output_filename, dpi=300)     
        
        result['status'] = 'SUCCESS'

    except Exception as e:
        
        result['status'] = 'ERROR'
        traceback.print_stack()
        print(e) 
    
    return result



print("Create output dir: %s" % (args.output_dir,))
Path(args.output_dir).mkdir(parents=True, exist_ok=True)


output_filename = os.path.join(args.output_dir, "wrf_budget_analysis_zidx-%d_%s.png" % (args.z_idx, time_beg.strftime("%Y-%m-%d_%H%M%S"), ))
plotBudget(time_beg, time_end, output_filename)
