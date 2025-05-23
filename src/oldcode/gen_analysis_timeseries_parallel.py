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
import wrf_load_helper 
import datetime
import os

cp_a  = 1004.0  # J / kg / K
Lq = 2.5e6      # J / kg
g0 = 9.81       # m / s^2

def integrateVertically(X, ds, avg=False):

    MUB = ds.MUB
    DNW = ds.DNW
    MU_FULL = ds.MU + ds.MUB
    MU_STAR = MU_FULL / MUB
    integration_factor = - MUB * DNW / g0  # notice that DNW is negative, so we need to multiply by -1

    X_STAR = X * MU_STAR
    X_INT = (integration_factor * X_STAR).sum(dim="bottom_top")

    if avg:
        sum_integration_factor = integration_factor.sum(dim="bottom_top")
        X_INT = X_INT / sum_integration_factor

    return X_INT


def onlyPos(x):
    return 0.0 if x < 0.0 else x



def genAnalysis(
    input_dir,
    output_filename,
    exp_beg_time,
    wrfout_data_interval,
    frames_per_wrfout_file,
    reltime_rngs,
    avg_before_analysis,
    x_rng,
):
    
    result = dict(output_filename=output_filename, status='UNKNOWN')

    try:
    
        exp_beg_time = pd.Timestamp(exp_beg_time)
    
        merge_data = []


        for i, reltime_rng in enumerate(reltime_rngs):
            
            selected_time_beg = exp_beg_time + reltime_rng[0]
            selected_time_end = exp_beg_time + reltime_rng[1]

            print("[%d] Analyzing time inverval: %s ~ %s" % (
                i,
                selected_time_beg.strftime("%Y-%m-%d_%H:%M:%S"), 
                selected_time_end.strftime("%Y-%m-%d_%H:%M:%S"),           
            ))
             
            ds = genAnalysis_subset(
                input_dir,
                exp_beg_time,
                wrfout_data_interval,
                frames_per_wrfout_file,
                reltime_rng,
                avg_before_analysis,
                x_rng,
            )
    
            merge_data.append(ds)



        print("Merging data...")
        new_ds = xr.merge(merge_data)

            
        full_range_time_beg = exp_beg_time + reltime_rngs[0][0]
        full_range_time_end = exp_beg_time + reltime_rngs[-1][1]
        new_ds.attrs["time_beg"] = full_range_time_beg.strftime("%Y-%m-%d_%H:%M:%S"),
        new_ds.attrs["time_end"] = full_range_time_end.strftime("%Y-%m-%d_%H:%M:%S"),
 
        print("Writing file: %s" % (output_filename,))
        new_ds.to_netcdf(
            output_filename,
            unlimited_dims="time",
            encoding={'time':{'units':'hours since 2001-01-01'}}
        )

        
        result['status'] = 'OK'

    
    except Exception as e:
        result['status'] = 'ERROR'
        traceback.print_exc()

    return result

def genAnalysis_subset(
    input_dir,
    exp_beg_time,
    wrfout_data_interval,
    frames_per_wrfout_file,
    reltime_rng,
    avg_before_analysis,
    x_rng,
):



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

    # Cannot use the following to get surface pressure:
    #PRES = ds.PB + ds.P
    #SFC_PRES = PRES.isel(bottom_top=0)
    
    # This is the correct one
    SFC_PRES = ds["PSFC"]

    PRES1000hPa=1e5

    R_over_cp = 2.0 / 7.0

    dT = (np.amax(ds["TSK"].to_numpy()) - np.amin(ds["TSK"].to_numpy())) / 2 

    TA = ( 300.0 + ds["T"].isel(bottom_top=0) ).rename("TA")
    TOA    = ( ds["TSK"] * (PRES1000hPa/SFC_PRES)**R_over_cp - TA ).rename("TOA")

    #  e1=svp1*exp(svp2*(tgdsa(i)-svpt0)/(tgdsa(i)-svp3)) 


    # Bolton (1980). But the formula is read from 
    # phys/physics_mmm/sf_sfclayrev.F90 Lines 281-285 (WRFV4.6.0)
    salinity_factor = 0.98
    E1 = 0.6112e3 * np.exp(17.67 * (ds["TSK"] - 273.15) / (ds["TSK"] - 29.65) ) * salinity_factor
    QSFCMR = (287/461.6) * E1 / (SFC_PRES - E1)
    
    QA  = ds["QVAPOR"].isel(bottom_top=0).rename("QA")

    QOA = QSFCMR - ds["QVAPOR"].isel(bottom_top=0)
    #QOA = xr.where(QOA > 0, QOA, 0.0)
    QOA = QOA.rename("QOA")
 
    #merge_data.append(WIND10)
    merge_data.append(TOA)
    merge_data.append(QOA)

    #merge_data.append( ( (ds["TSK"] - ds["T2"]) * WIND10 ).rename("WIND10TAO") )
 
    V_T = ds["V"]
    
    _tmp = ds["U"]
    U_T = ds["T"].copy().rename("U_T")

    #print("Dimesion of U_T: ")
    #print(U_T)

    U_T[:, :, :] = (_tmp.isel(west_east_stag=slice(1, None)).to_numpy() + _tmp.isel(west_east_stag=slice(0, -1)).to_numpy()) / 2
    U_T = U_T.rename("U_T")
    merge_data.append(U_T) 
 
    U_sfc = U_T.isel(bottom_top=0).to_numpy()
    V_sfc = V_T.isel(bottom_top=0)
    WND_sfc = (U_sfc**2 + V_sfc**2)**0.5
    WND_sfc = WND_sfc.rename("WND_sfc")

    HFX_from_FLHC = ds["FLHC"] * TOA
    QFX_from_FLQC = ds["FLQC"] * QOA
    LH_from_FLQC = Lq * QFX_from_FLQC

    HFX_from_FLHC = HFX_from_FLHC.rename("HFX_from_FLHC")
    QFX_from_FLQC = QFX_from_FLQC.rename("QFX_from_FLQC")
    LH_from_FLQC  = LH_from_FLQC.rename("LH_from_FLQC")

    merge_data.append(HFX_from_FLHC)
    merge_data.append(QFX_from_FLQC)
    merge_data.append(LH_from_FLQC)

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


    HFX_approx = C_H_WND_TOA + C_H_WND_cx_mul_TOA + WND_TOA_cx_mul_C_H + C_H_TOA_cx_mul_WND + C_H_WND_TOA_cx
    QFX_approx = C_Q_WND_QOA + C_Q_WND_cx_mul_QOA + WND_QOA_cx_mul_C_Q + C_Q_QOA_cx_mul_WND + C_Q_WND_QOA_cx
    LH_approx = Lq * QFX_approx
    
    HFX_approx = HFX_approx.rename("HFX_approx")
    QFX_approx = QFX_approx.rename("QFX_approx")
    LH_approx = LH_approx.rename("LH_approx")
 

    merge_data = []
    for varname in [
        "SWDOWN",
        "OLR",
        "HFX",
        "QFX",
        "LH",
    ]:
        _d = ds[varname].mean(dim="west_east")
        merge_data.append(_d)
        
        
  
    merge_data.extend([
        HFX_approx, QFX_approx, LH_approx,
        
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
 
        ds["RAINC"],        
        ds["RAINNC"],        
        ds["RAINSH"],        
        ds["SNOWNC"],        
        ds["HAILNC"],        
        ds["GRAUPELNC"],        
        
        ds["ACLHF"],        
        ds["ACHFX"],        
        
        ds["PBLH"],

        ds["HFX_from_FLHC"],
        ds["QFX_from_FLQC"],
        ds["LH_from_FLQC"],
    ])


    # Integrate water vapor, TKE

    for species in ["VAPOR", "CLOUD", "RAIN", "ICE", "SNOW"]:
        QX_varname = "Q%s" % (species,)
        QX_TTL_varname = "Q%s_TTL" % (species,)

        if not ( QX_varname in ds ):
            print("Variable %s not in file. Skip it." % (QX_varname,))
            continue

        QX_TTL = integrateVertically(ds[QX_varname], ds, avg=False).mean(dim="west_east").rename(QX_TTL_varname)
        merge_data.append(QX_TTL)

    THETA_MEAN = integrateVertically(300.0 + ds.T, ds, avg=True).mean(dim="west_east").rename("THETA_MEAN")
    merge_data.append(THETA_MEAN)

    # IWV
    IWV = integrateVertically(ds["QVAPOR"], ds, avg=False).rename("IWV")
 
    # IVT
    IVT_x = integrateVertically(ds["QVAPOR"] * U_T, ds, avg=False)
    IVT_y = integrateVertically(ds["QVAPOR"] * V_T, ds, avg=False)
    
    # code order matters here
    IVT = ((IVT_x**2 + IVT_y**2)**0.5).mean(dim="west_east").rename("IVT")
    IVT_x = IVT_x.mean(dim="west_east").rename("IVT_x")
    IVT_y = IVT_y.mean(dim="west_east").rename("IVT_y")

    merge_data.extend([IWV, IVT, IVT_x, IVT_y])

    if "QKE" in ds:
        TKE_TTL = integrateVertically(ds.QKE/2, ds, avg=False).mean(dim="west_east").rename("TKE_TTL")
        merge_data.append(TKE_TTL)


    # Merging data
    new_ds = xr.merge(merge_data)
    new_ds = new_ds.mean(dim='west_east', skipna=True, keep_attrs=True)
  
    if avg_before_analysis is False:
        new_ds = new_ds.mean(dim="time", skipna=True, keep_attrs=True).expand_dims(
            dim = {"time": [time_beg,]},
            axis = 0,
        )
        
    new_ds.attrs["dT"] = dT
    new_ds.attrs["time_beg"] = time_beg_str
    new_ds.attrs["time_end"] = time_end_str

    new_ds = new_ds.compute()

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
    parser.add_argument('--output-count', type=int, help="The numbers of output in a file.", default=1)
    parser.add_argument('--x-rng', type=float, nargs=2, help="X range in km", required=True)
    parser.add_argument('--avg-before-analysis', type=str, help="If set true, then the program will average first before analysis. This might affect the correlation terms.", choices=["TRUE", "FALSE"], required=True)
    parser.add_argument('--nproc', type=int, help="Number of parallel CPU.", default=1)
    
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
   
    print("Create dir: %s" % (args.output_dir,))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

 

    input_args = []
    
    number_of_intervals = (reltime_end - reltime_beg) / time_avg_interval
    if number_of_intervals % 1 != 0:
        raise Exception("Error: the interval of time_rng is not a multiple of time_avg_interval. Ratio: %.f" % (number_of_intervals,)) 

    number_of_intervals = int( np.ceil(number_of_intervals) ) 
    
    number_of_output_files = number_of_intervals / args.output_count
    if number_of_intervals % 1 != 0:
        raise Exception("Error: the number_of_intervals = %d is not a multiple of number_of_output_files = %d " % (
            number_of_intervals,
            number_of_output_files,
        ))
    number_of_output_files = int(number_of_output_files)
    
    print("There will be %d output files. Each contains %d data points" % (number_of_output_files, args.output_count,))
    
    time_interval_per_file = time_avg_interval * args.output_count
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
    
    
    
    
