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
    parser.add_argument('--input-dir-fmt', type=str, help='Input directories.', required=True)
    parser.add_argument('--output', type=str, help='Output filename in png.', default="")

    parser.add_argument('--Ugs', type=int, nargs="+", help='Ug in m/s.', required=True)
    parser.add_argument('--wnms', type=int, nargs="+", help='wnm in km.', required=True)
    parser.add_argument('--dSSTs', type=int, nargs="+", help='dSST in 0.01K.', required=True)

    parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
    parser.add_argument('--expand-time-min', type=int, help="Expanded time to load. This is combined with doing moving average.", default=0)
    parser.add_argument('--moving-avg-cnt', type=int, help="Number of points to do moving average.", default=1)
    parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
    parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
    parser.add_argument('--no-extra-variable', action="store_true", help='Number of frames in each wrfout file.',)


    args = parser.parse_args()

    print(args)

    exp_beg_time = pd.Timestamp(args.exp_beg_time)
    wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)
    time_beg = exp_beg_time + pd.Timedelta(hours=args.time_rng[0])
    time_end = exp_beg_time + pd.Timedelta(hours=args.time_rng[1])
    expanded_time = pd.Timedelta(minutes=args.expand_time_min)

    expanded_time_beg = time_beg - expanded_time 
    expanded_time_end = time_end + expanded_time 

    wsm = wrf_load_helper.WRFSimMetadata(
        start_datetime  = exp_beg_time,
        data_interval   = wrfout_data_interval,
        frames_per_file = args.frames_per_wrfout_file,
    )
    
    
    stat_array = ["mean", "std", ]

    data = None
   
    
    for i, Ug in enumerate(args.Ugs):
        for j, wnm in enumerate(args.wnms):
            for k, dSST in enumerate(args.dSSTs):
                
                sim_dir = os.path.join(
                    args.input_dir_fmt.format(
                        Ug = "%d" % ( Ug, ),
                        wnm = "%03d" % ( wnm, ),
                        dSST = "%03d" % ( int(dSST), ),
                    )
                )

                print("Loading wrf dir: %s" % (sim_dir,))
                _ds = wrf_load_helper.loadWRFDataFromDir(
                    wsm, 
                    sim_dir,
                    beg_time = time_beg,
                    end_time = time_end,
                    prefix="analysis_",
                    suffix=".nc",
                    avg=None,
                    verbose=False,
                    inclusive="left",
                )

                if not args.no_extra_variable:
                    PRECIP = _ds["RAINNC"] + _ds["RAINC"] + _ds["RAINSH"] + _ds["SNOWNC"] + _ds["HAILNC"] + _ds["GRAUPELNC"]
                    PRECIP = ( PRECIP - PRECIP.shift(time=1) ) / wrfout_data_interval.total_seconds()
                    PRECIP = PRECIP.rename("PRECIP")
                   
                    _ds = xr.merge([
                        _ds, PRECIP,
                    ])
     
                if data is None:

                    varnames = _ds.keys()

                    data = xr.Dataset(
                        data_vars = {
                            l : ( ["stat", "Ug", "wnm", "dSST"], np.zeros((len(stat_array), len(args.Ugs), len(args.wnms), len(args.dSSTs))) )
                            for l in varnames
                        },

                        coords = dict(
                            Ug = (["Ug"], args.Ugs),
                            wnm = (["wnm"], args.wnms),
                            dSST = (["dSST"], np.array(args.dSSTs)/100.0),
                            stat= (["stat",], stat_array),
                        )
                    )
                 
                
                # Do moving average
                _ds = _ds.rolling(time=args.moving_avg_cnt, center=True).mean()

                # Do uncertainties
                _ds_mean = _ds.mean(dim="time")
                _ds_std  = _ds.std(dim="time")
                
                for varname in data.keys():
                    data[varname][0, i, j, k] = _ds_mean[varname]
                    data[varname][1, i, j, k] = _ds_std[varname]
    
    
    print("Output file: ", args.output)
    data.to_netcdf(args.output)
    
    
