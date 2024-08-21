import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import datetime
import wrf_load_helper 
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-dirs', nargs="+", type=str, help='Input directories.', required=True)
parser.add_argument('--labels', nargs="+", type=str, help='Input directories.', required=True)
parser.add_argument('--varnames', type=str, nargs="+", help='Variable names.', required=True)
parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
parser.add_argument('--output', type=str, help='Output.', required=True)

args = parser.parse_args()

print(args)


exp_beg_time = pd.Timestamp(args.exp_beg_time)
wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)
time_beg = exp_beg_time + pd.Timedelta(hours=args.time_rng[0])
time_end = exp_beg_time + pd.Timedelta(hours=args.time_rng[1])

def relTimeInHrs(t):
    return (t - exp_beg_time.to_datetime64()) / np.timedelta64(1, 'h')

wsm = wrf_load_helper.WRFSimMetadata(
    start_datetime  = exp_beg_time,
    data_interval   = wrfout_data_interval,
    frames_per_file = args.frames_per_wrfout_file,
)

# Loading data
data = []
for i, input_dir in enumerate(args.input_dirs):

    print("Loading wrf dir: %s" % (input_dir,))
    ds = wrf_load_helper.loadWRFDataFromDir(
        wsm, 
        input_dir,
        beg_time = time_beg,
        end_time = time_end,
        prefix="analysis_",
        suffix=".nc",
        avg=None,
        verbose=False,
        inclusive="left",
    )

    TTL_RAIN = ds["RAINNC"] + ds["RAINC"]
    PRECIP = ( TTL_RAIN - TTL_RAIN.shift(time=1) ) / wrfout_data_interval.total_seconds()
    PRECIP = xr.where(np.isnan(PRECIP), 0.0, PRECIP)
    PRECIP = PRECIP.rename("PRECIP") 
    
    ds = xr.merge([ds, PRECIP])
    ds = xr.merge( [ ds[varname] for varname in args.varnames ] )

    data.append(ds)

stat_infos = dict(

    C_Q_WND_QOA_cx = dict(
        factor = 2.5e6,
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
    ),

    WND_QOA_cx_mul_C_Q = dict(
        factor = 2.5e6,
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
    ),

    C_Q_QOA_cx_mul_WND = dict(
        factor = 2.5e6,
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
    ),

    C_Q_WND_cx_mul_QOA = dict(
        factor = 2.5e6,
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
    ),

    C_Q_WND_QOA = dict(
        factor = 2.5e6,
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
    ),

    C_H_WND_TOA = dict(
        factor = 1.0,
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
    ),

    WND_TOA_cx_mul_C_H = dict(
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
    ),


    C_H_WND_TOA_cx = dict(
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
    ),

    C_H_TOA_cx_mul_WND = dict(
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
    ),

    C_H_WND_cx_mul_TOA = dict(
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
    ),

    PRECIP = dict(
        factor = 86400.0,
        unit = "$ \\mathrm{mm} / \\mathrm{day} $",
    ),

    HFX = dict(
        factor = 1,
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
    ),

    LH = dict(
        factor = 1,
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
    ),

 
    HFX_approx = dict(
        factor = 1,
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
    ),

    LH_approx = dict(
        factor = 1,
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
    ),


)



print("Doing statistics")

df_data = dict(
    label = [],
)

for varname in args.varnames:
    df_data["%s_mean" % varname] = []
    df_data["%s_std" % varname]  = []

for i, ds in enumerate(data):

    print("# Doing the %d-th data. Label : %s" % (i, args.labels[i]))

        
    df_data["label"].append(args.labels[i])
    for j, varname in enumerate(args.varnames):

        stat_info = stat_infos[varname]
        factor = stat_info["factor"] if "factor" in stat_info else 1.0
        d = ds[varname] * factor
        d = d.to_numpy()
        
        df_data["%s_mean" % varname].append(np.mean(d))
        df_data["%s_std"  % varname].append(np.std(d)) 

 
        #print('{varname:s} : {mean:f} (+- {std:f})'.format(
        #print('{varname:s} : {std:f} )'.format(
        #    varname = varname,
        #    mean = result['mean'],
        #    std  = result['std'],
        #))
 
        print('%s : %f (+- %f)' % (varname,
            df_data['%s_mean' % varname][-1],
            df_data['%s_std' % varname][-1],
        ))


df = pd.DataFrame(df_data)
print(df) 

if args.output != "":
   
    print("Output to csv file: ", args.output) 
    df.to_csv(args.output)



