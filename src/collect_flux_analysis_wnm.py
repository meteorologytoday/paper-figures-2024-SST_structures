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

    parser.add_argument('--Us', type=int, nargs="+", help='U in m/s.', required=True)
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
   
    
    for i, U in enumerate(args.Us):
        for j, wnm in enumerate(args.wnms):
            for k, dSST in enumerate(args.dSSTs):


                _wnm = wnm
                _dSST = dSST

                if dSST == 0:
                    if wnm != 0:
                        print("WARNING: dSST = 0, force wnm = 0")
                        _wnm = 0
 
                if wnm == 0:
                    if dSST != 0:
                        print("WARNING: wnm = 0, force dSST = 0")
                        _dSST = 0
                
                sim_dir = os.path.join(
                    args.input_dir_fmt.format(
                        U = "%d" % ( U, ),
                        wnm = "%03d" % ( _wnm, ),
                        dSST = "%03d" % ( int(_dSST), ),
                    )
                )

                try:
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

                    if "PRECIP" in _ds:
                        print("Drop PRECIP")
                        _ds = _ds.drop_vars(["PRECIP",])

                    if not args.no_extra_variable:
                        print("Compute PRECIP")
                        PRECIP = ( _ds["TTL_RAIN"] - _ds["TTL_RAIN"].shift(time=1) ) / wrfout_data_interval.total_seconds()
                        PRECIP = PRECIP.rename("PRECIP")
                       
                        _ds = xr.merge([
                            _ds, PRECIP,
                        ])
         
                    if data is None:

                        varnames = _ds.keys()
                        
                        data = xr.Dataset(
                            data_vars = {
                                l : ( ["stat", "U", "wnm", "dSST"], np.zeros((len(stat_array), len(args.Us), len(args.wnms), len(args.dSSTs))) )
                                for l in varnames
                            },

                            coords = dict(
                                U = (["U"], args.Us),
                                wnm = (["wnm"], args.wnms),
                                dSST = (["dSST"], np.array(args.dSSTs)/100.0),
                                stat= (["stat",], stat_array),
                            )
                        )
                     
                    
                    # Do moving average
                    _ds = _ds.rolling(time=args.moving_avg_cnt, center=True).mean()

                    if "west_east" in _ds.dims:
                        _ds = _ds.mean(dim="west_east")

                    # Do uncertainties
                    _ds_mean = _ds.mean(dim="time")
                    _ds_std  = _ds.std(dim="time")
                    
                    for varname in data.keys():
                        print("Saving varname: ", varname)
                        data[varname][0, i, j, k] = _ds_mean[varname]
                        data[varname][1, i, j, k] = _ds_std[varname]
        
                except Exception as e:
                    
                    import traceback
                    
                    print("Exception happens.")
                    traceback.print_exc()

                    
                     
    print("Output file: ", args.output)
    data.to_netcdf(args.output)
    
    
