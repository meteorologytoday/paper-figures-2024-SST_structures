import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import wrf_load_helper 
import datetime
import os
import wrf_preprocess

varinfos = dict(

    SST = dict(
        selector = None,
        wrf_varname = "TSK",
        unit = "K",
        label = "SST",
    ), 


    TA = dict(
        selector = dict(bottom_top=0),
        wrf_varname = "T",
        label = "$\\Theta_{A}$",
        unit = "K",
    ), 

    TOA = dict(
        wrf_varname = "TOA",
        label = "$\\Theta_{OA}$",
        unit = "K",
    ), 

    QOA = dict(
        wrf_varname = "QOA",
        label = "$Q_{OA}$",
        unit = "g / kg",
    ), 

    CH = dict(
        wrf_varname = "CH",
        label = "$C_{H}$",
    ), 

    CQ = dict(
        wrf_varname = "CQ",
        label = "$C_{Q}$",
    ), 


    UA = dict(
        selector = dict(bottom_top=0),
        wrf_varname = "U",
        label = "$u_{A}$",
        unit = "$ \\mathrm{m} \\, / \\, \\mathrm{s}$",
    ), 

    VA = dict(
        selector = dict(bottom_top=0),
        wrf_varname = "V",
        label = "$v_{A}$",
        unit = "$ \\mathrm{m} \\, / \\, \\mathrm{s}$",
    ), 

    VORA = dict(
        selector = dict(bottom_top=0),
        wrf_varname = "VOR",
        label = "$\\zeta_{A}$",
        unit = "$\\mathrm{s}^{-1}$",
    ), 

    DIVA = dict(
        selector = dict(bottom_top=0),
        wrf_varname = "DIV",
        label = "$D_{A}$",
        unit = "$\\mathrm{s}^{-1}$",
    ), 





)

def corr(arrA, arrB):
    
    arrA_anom = arrA - np.mean(arrA)
    arrB_anom = arrB - np.mean(arrB)

    r = np.sum(arrA_anom * arrB_anom) / (np.sum(arrA_anom**2.0)*np.sum(arrB_anom**2.0))**0.5

    return r


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
    parser.add_argument('--ctl-varname', type=str, help="The varname to be the forcing.", required=True)
    parser.add_argument('--varnames', type=str, nargs="+", help="Varnames to do the analysis.", required=True)
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
    stats = ["coherence_C", "coherence_Q", "coherence_S1", "coherence_S2", "cross_corr", "fourier_real", "fourier_imag"]

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

            ds_base = xr.merge([
                ds_base,
                wrf_preprocess.genAnalysis(ds_base, wsm.data_interval),
            ]).mean(dim="time")


        Nx = len(ds_base.coords["west_east"])
        Lx = DX * Nx
        Ls[i] = Lx / tracking_wnm
        X_sU = DX * np.arange(Nx+1)
        X_sT = (X_sU[1:] + X_sU[:-1]) / 2
        freq = np.fft.fftfreq(Nx, d=DX)

        freq_N = Nx // 2
        
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
            
        ds = xr.merge([
            ds,
            wrf_preprocess.genAnalysis(ds, wsm.data_interval),
        ])

        # Cannot initialize until knowing how many time points there are
        if data is None: 

            data = {
                measure : {
                    varname : np.zeros( (ds.dims["time"], len(args.tracking_wnms)), ) for varname in args.varnames
                } for measure in stats
            }


        for t in range(ds.dims["time"]):
        
            _ds = ds.isel(time=t)
 
            d_anom = dict()
            for varname in args.varnames + [args.ctl_varname,]:

                if varname in d_anom:
                    print("Varname %s is already loaded. Skip." % (varname,))
                    continue

                varinfo = varinfos[varname]
                
                selector = varinfo["selector"] if "selector" in varinfo else None
                wrf_varname = varinfo["wrf_varname"] if "wrf_varname" in varinfo else varname

                da_base = ds_base[wrf_varname]
                da = _ds[wrf_varname]

                if selector is not None:
                    da_base = da_base.isel(**selector)
                    da      = da.isel(**selector)
                 
                if "south_north" in da.dims:
                    da = da.isel(south_north=0)
                    da_base = da_base.isel(south_north=0)
                
                elif "south_north_stag" in da.dims:
                    da = da.isel(south_north_stag=0)
                    da_base = da_base.isel(south_north_stag=0)

                dvar = da - da_base
                dvar = dvar.to_numpy()

                if varname == "UA":
                    dvar = ( dvar[1:] + dvar[:-1] ) / 2
     
                # Compute transfer function
                d_anom[varname] = dvar



               
            X0 = d_anom[args.ctl_varname]
            sp_X0 = np.fft.fft(X0)[tracking_wnm]
             
            for varname in args.varnames:
                
                print("Doing coherence analysis of %s" % (varname,)) 
                X1 = d_anom[varname]
                sp_X1 = np.fft.fft(X1)[tracking_wnm]
                
                #G_X0X1 = np.real(sp_X0 * np.conjugate(sp_X1))
                S_X0X1 = np.conjugate(sp_X0) * sp_X1
                S_X0X0 = np.abs(sp_X0)**2
                S_X1X1 = np.abs(sp_X1)**2
                
                c_real = np.real(sp_X1)
                c_imag = np.imag(sp_X1)
               

                coherence = np.abs(S_X0X1)**2 / (S_X0X0 * S_X1X1)

                #print("S_X0X0: ", S_X0X0)
                #print("S_X1X1: ", S_X1X1)
                #print("S_X0X1: ", S_X0X1)
                #print("coherence: ", coherence)

                data["coherence_C"][varname][t, i] = np.real(S_X0X1)
                data["coherence_Q"][varname][t, i] = np.imag(S_X0X1)
                data["coherence_S1"][varname][t, i] = S_X0X0
                data["coherence_S2"][varname][t, i] = S_X1X1
                data["cross_corr"][varname][t, i] = corr(X0, X1)
                data["fourier_real"][varname][t, i] = c_real
                data["fourier_imag"][varname][t, i] = c_imag
   

    print("Rearranging data for output...")        
    for varname in args.varnames:
        
        _tmp = []
        for stat in stats:
            _tmp.append(data[stat][varname])

        stacked = np.stack(_tmp, axis=0)
        data_vars[varname] = ( ["stat", "time", "wvlen" ], stacked )

    new_ds = xr.Dataset(
        data_vars = data_vars,
        coords = dict(
            time = ds.coords["time"],
            wvlen = Ls,
            stat = stats,
        ),
    )

    new_ds = new_ds.transpose("time", "wvlen", "stat")

    print("Output file: ", args.output) 
    new_ds.to_netcdf(args.output, unlimited_dims="time")
