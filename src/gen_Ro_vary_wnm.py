import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import wrf_load_helper 
import datetime
import os
import wrf_preprocess

g = 9.81

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-dirs', type=str, nargs="+", help='Input directories.', required=True)
    parser.add_argument('--input-dirs-base', type=str, nargs="+", help='Input directories.', required=True)
    parser.add_argument('--tracking-wnms', type=int, nargs="+", help='The wave number to trace.', required=True)
    parser.add_argument('--output', type=str, help='Output filename in png.', required=True)
    parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
    parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
    parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
    parser.add_argument('--wrfout-suffix', type=str, default="")
    parser.add_argument('--avg-time-in-seconds', type=float, default=3600)



    args = parser.parse_args()

    print(args)

    if len(args.tracking_wnms) != len(args.input_dirs):
        raise Exception("Length of `--tracking-wnms` (%d) does not equal to length of `--input-dirs` (%d). " % (
            len(args.tracking_wnms),
            len(args.input_dirs),
        ))


    same_base = False
    if len(args.input_dirs_base) == 1:
       
        args.input_dirs_base = [ args.input_dirs_base[0] ] * len(args.input_dirs) 


    if np.all( [ input_dir_base == args.input_dirs_base[0] for input_dir_base in args.input_dirs_base  ] ):
        same_base = True
        print("# same_base = ", same_base)


    if len(args.input_dirs_base) != len(args.input_dirs):
        
        raise Exception("Length of `--input-dirs-base` (%d) does not equal to length of `--input-dirs` (%d). " % (
            len(args.input_dirs_base),
            len(args.input_dirs),
        ))

    # Used for final output
    data_vars = dict()
    
    tracking_wnms = np.array(args.tracking_wnms, dtype=int)
 
    exp_beg_time = pd.Timestamp(args.exp_beg_time)
    wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)
    time_beg = exp_beg_time + pd.Timedelta(hours=args.time_rng[0])
    time_end = exp_beg_time + pd.Timedelta(hours=args.time_rng[1])
    avg_time = pd.Timedelta(seconds=args.avg_time_in_seconds)


    wsm = wrf_load_helper.WRFSimMetadata(
        start_datetime  = exp_beg_time,
        data_interval   = wrfout_data_interval,
        frames_per_file = args.frames_per_wrfout_file,
    )
    
    # Loading     
    data = None
    Ls = np.zeros( (len(args.tracking_wnms), ) )
    for i in range(len(args.input_dirs)):
        
        input_dir_base = args.input_dirs_base[i] 
        input_dir      = args.input_dirs[i]
        tracking_wnm   = tracking_wnms[i] 
        print("Loading base wrf dir: %s" % (input_dir_base,))

        if i == 0 or not same_base:
            ds_base = wrf_load_helper.loadWRFDataFromDir(
                wsm, 
                input_dir_base,
                beg_time = time_beg,
                end_time = time_end,
                suffix=args.wrfout_suffix,
                avg=None,
                verbose=False,
                inclusive="left",
            )

            DX = ds_base.attrs["DX"]
            Nx = len(ds_base.coords["west_east"])

            ds_base = ds_base.mean(dim=["time", "west_east", "west_east_stag", "south_north_stag", "south_north"])

        Lx = DX * Nx
        Ls[i] = Lx / tracking_wnm
        freq_N = Nx // 2
        
        Z_W = ( ( ds_base["PHB"] + ds_base["PH"] ) / g ).to_numpy()
        Z_T = ( Z_W[1:] + Z_W[:-1] ) / 2.0

        print("Loading the %d-th wrf dir: %s" % (i, input_dir,))

        ds = wrf_load_helper.loadWRFDataFromDir(
            wsm, 
            input_dir,
            beg_time = time_beg,
            end_time = time_end,
            suffix=args.wrfout_suffix,
            avg=None,
            verbose=False,
            inclusive="left",
        )
        ds = ds.mean(dim=["south_north", "south_north_stag"])

        # Cannot initialize until knowing how many time points there are
        if data is None: 
            data = np.zeros( (ds.dims["time"], ds.dims["bottom_top"], len(args.tracking_wnms)), )

        
        diff = ds - ds_base

        dU = diff["U"].std(dim="west_east_stag")
        F = ds["F"].mean(dim="west_east")
        
        Ro = dU / (F * Ls[i])
        #Ro = Ro.mean(dim="time")
        print(Ro) 
        data[:, :, i] = Ro.to_numpy()
        

    print("Rearranging data for output...")        
    data_vars = dict(Ro = ( ["time", "bottom_top", "wvlen" ], data) )

    new_ds = xr.Dataset(
        data_vars = dict(
            Ro = ( ["time", "Z", "wvlen" ], data),
        ),
        coords = dict(
            time = ds.coords["time"],
            wvlen = Ls,
            Z = (["Z",], Z_T),
        ),
    )

    new_ds = new_ds.transpose("time", "wvlen", "Z")

    print("Output file: ", args.output) 
    new_ds.to_netcdf(args.output, unlimited_dims="time")
