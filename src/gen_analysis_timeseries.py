import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import wrf_load_helper 
import datetime
import os

cp_a  = 1004.0  # J / kg / K
Lq = 2.5e6

def genAnalysis(
    input_dir,
    exp_beg_time,
    wrfout_data_interval,
    frames_per_wrfout_file,
    reltime_rng,
    avg_before_analysis,
    x_rng,
):

    exp_beg_time = pd.Timestamp(exp_beg_time)
    time_beg = exp_beg_time + reltime_rng[0]
    time_end = exp_beg_time + reltime_rng[1]
    
    time_beg_str = time_beg.strftime("%Y-%m-%dT%H:%M:%S")
    time_end_str = time_end.strftime("%Y-%m-%dT%H:%M:%S")
    
    time_bnd = xr.DataArray(
        name="time_bnd",
        data=[time_beg, time_end],
        dims=["time_bnd",],
        coords=dict(
            
        ),
        attrs=dict(
            reference_time=pd.Timestamp("1970-01-01"),
        ),
    )

    def relTimeInHrs(t):
        return (t - exp_beg_time.to_datetime64()) / np.timedelta64(1, 'h')


    data_interval = pd.Timedelta(seconds=wrfout_data_interval)

    # Loading data
        
    print("Loading directory: %s" % (input_dir,))

    wsm = wrf_load_helper.WRFSimMetadata(
        start_datetime = exp_beg_time,
        data_interval = data_interval,
        frames_per_file = frames_per_wrfout_file,
    )

    ds = wrf_load_helper.loadWRFDataFromDir(
        wsm, 
        input_dir,
        beg_time = time_beg,
        end_time = time_end,
        prefix="wrfout_d01_",
        avg="ALL" if avg_before_analysis else None,
        verbose=False,
        inclusive="both",
    )

    print("Processing...")
    ds = ds.mean(dim=['south_north', 'south_north_stag'], keep_attrs=True)
    
    ref_ds = ds.mean(dim=['time'], keep_attrs=True)
    Nx = ref_ds.dims['west_east']
    Nz = ref_ds.dims['bottom_top']

    X_sU = ref_ds.DX * np.arange(Nx+1) / 1e3
    X_sT = (X_sU[1:] + X_sU[:-1]) / 2
    X_T = np.repeat(np.reshape(X_sT, (1, -1)), [Nz,], axis=0)
    X_W = np.repeat(np.reshape(X_sT, (1, -1)), [Nz+1,], axis=0)

    Z_W = (ref_ds.PHB + ref_ds.PH) / 9.81
    Z_T = (Z_W[1:, :] + Z_W[:-1, :]) / 2

    ds = ds.assign_coords(dict(
        west_east = X_sT, 
        west_east_stag = X_sU, 
    ))

    merge_data = [ds,]

    dT = (np.amax(ds["TSK"].to_numpy()) - np.amin(ds["TSK"].to_numpy())) / 2 

    TA = ( 300.0 + ds["T"].isel(bottom_top=0) ).rename("TA")
    TOA    = ( ds["TSK"] - TA ).rename("TOA")

    # Bolton (1980)
    E1 = 0.6112e3 * np.exp(17.67 * (ds["TSK"] - 273.15) / (ds["TSK"] - 29.65) )
    QSFCMR = 0.622 * E1 / (ds["PSFC"] - E1)
    QA  = ds["QVAPOR"].isel(bottom_top=0).rename("QA")
    QOA = QSFCMR - ds["QVAPOR"].isel(bottom_top=0)
    QOA = QOA.rename("QOA")
 
    #merge_data.append(WIND10)
    merge_data.append(TOA)
    merge_data.append(QOA)

    #merge_data.append( ( (ds["TSK"] - ds["T2"]) * WIND10 ).rename("WIND10TAO") )
 
    _tmp = ds["U"]
    U_T = ds["T"].copy()
    U_T[:, :] = (_tmp.isel(west_east_stag=slice(1, None)).to_numpy() + _tmp.isel(west_east_stag=slice(0, -1)).to_numpy()) / 2
    U_T = U_T.rename("U_T")
    merge_data.append(U_T) 
 
    U_sfc = U_T.isel(bottom_top=0).to_numpy()
    V_sfc = ds["V"].isel(bottom_top=0)
    WND_sfc = (U_sfc**2 + V_sfc**2)**0.5
    WND_sfc = WND_sfc.rename("WND_sfc")


    C_H = ds["FLHC"] / WND_sfc
    C_H = C_H.rename("C_H")
    
    C_Q = ds["FLQC"] / WND_sfc
    C_Q = C_Q.rename("C_Q")

    merge_data.append(WND_sfc)
    merge_data.append(C_H)
    merge_data.append(C_Q)
   
    ds = xr.merge(merge_data)

    if x_rng is not None:
        ds = ds.where(
            (ds.coords["west_east"] >= x_rng[0]) & 
            (ds.coords["west_east"] <= x_rng[1]) 
        )
        
    # Surface flux approximation comes from 
    # the file : module_sf_mynn.F    
    #Ulev1 = ds["U_T"].isel(bottom_top=0).to_numpy()
    #Vlev1 = ds["V"].isel(bottom_top=0).to_numpy()
    

    HFX_approx = ds["C_H"] * ds["WND_sfc"] * ds["TOA"]
    HFX_approx = HFX_approx.rename("HFX_approx")
    
    QFX_approx = ds["C_Q"] * ds["WND_sfc"] * ds["QOA"]
    _tmp = QFX_approx.to_numpy()
    _tmp[_tmp <= -0.02] = -0.02
    QFX_approx[:] = _tmp
    LH_approx = Lq * QFX_approx
    LH_approx = LH_approx.rename("LH_approx")
   
    ds = xr.merge([ds, HFX_approx, LH_approx])
    
    WND_sfc_mean = ds["WND_sfc"].mean(dim="west_east").rename("WND_sfc_mean")

    def horDecomp(da, name_m="mean", name_p="prime"):
        m = da.mean(dim="west_east").rename(name_m)
        p = (da - m).rename(name_p) 
        return m, p

    WND_m, WND_p = horDecomp(ds["WND_sfc"], "WND_m", "WND_p")
    
    C_H_m, C_H_p = horDecomp(ds["C_H"], "C_H_m", "C_H_p")
    TOA_m, TOA_p = horDecomp(ds["TOA"], "TOA_m", "TOA_p")
    
    C_H_TOA_cx = (C_H_p * TOA_p).mean(dim="west_east").rename("C_H_TOA_cx")
    C_H_WND_cx = (C_H_p * WND_p).mean(dim="west_east").rename("C_H_WND_cx")
    WND_TOA_cx = (WND_p * TOA_p).mean(dim="west_east").rename("WND_TOA_cx")
    C_H_WND_TOA_cx = (C_H_p * WND_p * TOA_p).mean(dim="west_east").rename("C_H_WND_TOA_cx")

    C_H_WND_TOA        = (C_H_m * WND_m * TOA_m).rename("C_H_WND_TOA")
    C_H_WND_cx_mul_TOA = (C_H_WND_cx * TOA_m).rename("C_H_WND_cx_mul_TOA")
    WND_TOA_cx_mul_C_H = (WND_TOA_cx * C_H_m).rename("WND_TOA_cx_mul_C_H")
    C_H_TOA_cx_mul_WND = (C_H_TOA_cx * WND_m).rename("C_H_TOA_cx_mul_WND")
    F_SEN_res = (
        (C_H_WND_TOA + C_H_WND_cx_mul_TOA + WND_TOA_cx_mul_C_H + C_H_TOA_cx_mul_WND + C_H_WND_TOA_cx)
        - HFX_approx
    ).rename("F_SEN_res")
   


    C_Q_m, C_Q_p = horDecomp(ds["C_Q"], "C_Q_m", "C_Q_p")
    QOA_m, QOA_p = horDecomp(ds["QOA"], "QOA_m", "QOA_p")

    C_Q_QOA_cx = (C_Q_p * QOA_p).mean(dim="west_east").rename("C_Q_QOA_cx")
    C_Q_WND_cx = (C_Q_p * WND_p).mean(dim="west_east").rename("C_Q_WND_cx")
    WND_QOA_cx = (WND_p * QOA_p).mean(dim="west_east").rename("WND_QOA_cx")
    C_Q_WND_QOA_cx = (C_Q_p * WND_p * QOA_p).mean(dim="west_east").rename("C_Q_WND_QOA_cx")

    C_Q_WND_QOA        = (C_Q_m * WND_m * QOA_m).rename("C_Q_WND_QOA")
    C_Q_WND_cx_mul_QOA = (C_Q_WND_cx * QOA_m).rename("C_Q_WND_cx_mul_QOA")
    WND_QOA_cx_mul_C_Q = (WND_QOA_cx * C_Q_m).rename("WND_QOA_cx_mul_C_Q")
    C_Q_QOA_cx_mul_WND = (C_Q_QOA_cx * WND_m).rename("C_Q_QOA_cx_mul_WND")


    HFX = ds["HFX"].mean(dim="west_east")
    LH  = ds["LH"].mean(dim="west_east")

    """
    TTL_RAIN = ds["RAINNC"] + ds["RAINC"]
    dt = ds.coords["time"]
    PRECIP = ( xr.shift(TTL_RAIN, time=1) - TTL_RAIN ) / ()
    
    PRECIP[1:] = (tmp[1:] - tmp[:-1]) / coarse_grained_time.total_seconds()
    PRECIP = PRECIP.rename("PRECIP") 
    """

    new_ds = xr.merge([
        HFX, LH, 
        HFX_approx, LH_approx,
        
        WND_m, WND_p,
        TA, QA,       
 
        C_H_m, C_H_p,
        TOA_m, TOA_p,
        C_H_TOA_cx,
        C_H_WND_cx,
        WND_TOA_cx,
        C_H_WND_TOA_cx,
        
        C_Q_m, C_Q_p,
        QOA_m, QOA_p,
        C_Q_QOA_cx,
        C_Q_WND_cx,
        WND_QOA_cx,
        C_Q_WND_QOA_cx,

        C_H_WND_TOA,
        C_H_WND_cx_mul_TOA,
        WND_TOA_cx_mul_C_H,
        C_H_TOA_cx_mul_WND,

        C_Q_WND_QOA,
        C_Q_WND_cx_mul_QOA,
        WND_QOA_cx_mul_C_Q,
        C_Q_QOA_cx_mul_WND,
 
        F_SEN_res,

        ds["RAINC"],        
        ds["RAINNC"],        
    
        ds["PBLH"],    
    ])

    new_ds = new_ds.mean(dim='west_east', skipna=True, keep_attrs=True)
  
    if avg_before_analysis is False:
        new_ds = new_ds.mean(dim="time", skipna=True, keep_attrs=True).expand_dims(
            dim = {"time": [selected_time_beg,]},
            axis = 0,
        )
        
    new_ds.attrs["dT"] = dT
    new_ds.attrs["time_beg"] = time_beg_str
    new_ds.attrs["time_end"] = time_end_str

    ds.close()

    return new_ds



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-dir', type=str, help='Input directories.', required=True)
    parser.add_argument('--output-dir', type=str, help='Output directory.', required=True)
    parser.add_argument('--exp-beg-time', type=str, help='Experiment begin time.', required=True)
    parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
    parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)

    parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
    parser.add_argument('--time-avg-interval', type=int, help="The interval of time to do the average. Unit is in minutes. If it is not specified or 0 then all data is averaged.", default=0)
    parser.add_argument('--x-rng', type=float, nargs=2, help="X range in km", required=True)
    parser.add_argument('--avg-before-analysis', type=str, help="If set true, then the program will average first before analysis. This might affect the correlation terms.", choices=["TRUE", "FALSE"], required=True)
    
    args = parser.parse_args()

    print(args)
   
    avg_before_analysis = True if args.avg_before_analysis == "TRUE" else False 
  
    x_rng = None #np.array(args.x_rng) * 1e3
    print(x_rng)


    # Decide the time decomposition
    print("Processing ...")

    exp_beg_time = pd.Timestamp(args.exp_beg_time)
    reltime_beg = pd.Timedelta(hours=args.time_rng[0])
    reltime_end = pd.Timedelta(hours=args.time_rng[1])
    time_avg_interval = pd.Timedelta(minutes=args.time_avg_interval)
    
    if time_avg_interval / pd.Timedelta(seconds=1) == 0:  # if not specified or 0
        print("The parameter `--time-avg-interval` is zero, assume the average is the entire interval.")
        time_avg_interval = time_end - time_beg
    
    
    number_of_intervals = (reltime_end - reltime_beg) / time_avg_interval
    if number_of_intervals % 1 != 0:
        print("Warning: the interval of time_rng is not a multiple of time_avg_interval. Ratio: %.f" % (number_of_intervals,)) 

    number_of_intervals = int( np.ceil(number_of_intervals) ) 
    for i in range(number_of_intervals):

        selected_time_beg = exp_beg_time      + time_avg_interval * i
        selected_time_end = selected_time_beg + time_avg_interval

        selected_reltime_beg = selected_time_beg - exp_beg_time
        selected_reltime_end = selected_time_end - exp_beg_time

        print("Analyzing time inverval: %s ~ %s" % (
            selected_time_beg.strftime("%Y-%m-%d_%H:%M:%S"), 
            selected_time_end.strftime("%Y-%m-%d_%H:%M:%S"),           
        ))

        _ds = genAnalysis(
            args.input_dir,
            args.exp_beg_time,
            args.wrfout_data_interval,
            args.frames_per_wrfout_file,
            [selected_reltime_beg, selected_reltime_end],
            avg_before_analysis,
            x_rng,
        )
        
        filename = os.path.join(
            args.output_dir,
            "analysis_{timestr:s}.nc".format(
                timestr = selected_time_beg.strftime("%Y-%m-%d_%H:%M:%S"),
            )
        )
        print("[%d] Writing file: %s" % (i, filename,))
        _ds.to_netcdf(
            filename,
            unlimited_dims="time",
        )
    
    print("Done")
    
    
    
    
