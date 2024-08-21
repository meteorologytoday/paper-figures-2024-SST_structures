import traceback
from multiprocessing import Pool
import multiprocessing
from pathlib import Path
import os



import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import datetime
import wrf_load_helper 
import cmocean
import colorblind


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-dir', type=str, help='Input directory.', required=True)
parser.add_argument('--output-dir', type=str, help='Output filename in png.', required=True)
parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)

parser.add_argument('--avg-interval', type=int, help='The averge time interval in minutes', required=True)
parser.add_argument('--output-frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
parser.add_argument('--overwrite', action="store_true", help='Whether to overwrite the output.')

args = parser.parse_args()

print(args)

exp_beg_time = pd.Timestamp(args.exp_beg_time)
wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)
time_beg = exp_beg_time + pd.Timedelta(minutes=args.time_rng[0])
time_end = exp_beg_time + pd.Timedelta(minutes=args.time_rng[1])

avg_interval = pd.Timedelta(minutes=args.avg_interval)

print("Average interval: ", str(avg_interval))

number_of_output = (time_end - time_beg) / avg_interval
snapshots_per_avg = avg_interval / wrfout_data_interval
number_of_output_files = np.ceil(number_of_output_files / args.output_frames_per_wrfout_file)

if number_of_output % 1 != 0:
    raise Exception("`--time-rng` is not a multiple of `--avg-interval`")

if avg_interval % 1 != 0:
    raise Exception("`--avg-interval` is not a multiple of `--wrfout-data-interval`")

print("Expected total output: %d files, %d data per file, total %d data. "  %  (number_of_output_files, args.output_frames_per_wrfout_file, number_of_output))


def work(
    detail,
):

    result = dict(status="UNKNOWN")

    try:
        wsm = wrf_load_helper.WRFSimMetadata(
            start_datetime  = detail["exp_beg_time"],
            data_interval   = detail["wrfout_data_interval"],
            frames_per_file = detail["frames_per_wrfout_file"],
        )

        ds = wrf_load_helper.loadWRFDataFromDir(
            wsm, 
            detail["input_dir"],
            beg_time = detail["time_beg"],
            end_time = detail["time_end"],
            prefix=detail["input_prefix"],
            avg = detail["avg_interval"],
            verbose=False,
            inclusive="left",
        )
        
        
        output_filename = "{prefix:s}_{timestr:s}".format(
            prefix = detail["output_prefix"],
            beg_time.strftime("%Y-%m-%d_%H:%M:%S")
        )
            
        output_full_filename = os.path.join(
            detail["output_dir"],
            output_filename,
        )


        print("Output file: ", output_full_filename)
        ds.to_netcdf(
            output_full_filename,
            unlimited_dims="time",
        ) 
   
    except Exception as e:
        result['status'] = 'ERROR'
        traceback.print_exc()

    return result


for i in range(number_of_output_files):

    reltime_rngs = []
    reltime_beg_of_file = reltime_beg + i * time_interval_per_file
    time_beg_of_file = exp_beg_time + reltime_beg_of_file

    for j in range(args.output_count):
        
        reltime_rngs.append(
            [ 
                reltime_beg_of_file + j     * time_avg_interval,
                reltime_beg_of_file + (j+1) * time_avg_interval,
            ]
        )

    filename = os.path.join(
        args.output_dir,
        "analysis_{timestr:s}.nc".format(
            timestr = time_beg_of_file.strftime("%Y-%m-%d_%H:%M:%S"),
        )
    )

    if os.path.exists(filename):
        print("File %s already exists. Skip it." % (filename,))
        continue

   
    input_args.append(
        (
            args.input_dir,
            filename,
            args.exp_beg_time,
            args.wrfout_data_interval,
            args.frames_per_wrfout_file,
            reltime_rngs,
            avg_before_analysis,
            x_rng,
        )
    )



failed_files = []
with Pool(processes=args.nproc) as pool:

    results = pool.starmap(genAnalysis, input_args)

    for i, result in enumerate(results):
        if result['status'] != 'OK':
            print('!!! Failed to generate output : %s.' % (result['output_filename'],))
            failed_files.append(result['output_filename'])


print("Tasks finished.")

print("Failed files: ")
for i, failed_file in enumerate(failed_files):
    print("%d : %s" % (i+1, failed_file,))


print("Done")



