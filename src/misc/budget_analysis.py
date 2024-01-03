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

# Air density
Rd = 287.053 # J / K / kg
cp = 1004    # J / K / kg
T0 = 300.0 # Base theta_0  (K)
p0 = 1e5 # Reference pressure (Pa)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-dir', type=str, help='Input directories.', required=True)
parser.add_argument('--output-dir', type=str, help='Output filename in png.', required=True)
parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
parser.add_argument('--nproc', type=int, help='Number of processes.', default=1)
parser.add_argument('--time-rng', type=int, nargs=2, help="Time range to process in minutes after --exp-beg-time", required=True)

args = parser.parse_args()

print(args)

exp_beg_time = pd.Timestamp(args.exp_beg_time)
wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)

time_beg = exp_beg_time + pd.Timedelta(minutes=args.time_rng[0])
time_end = exp_beg_time + pd.Timedelta(minutes=args.time_rng[1])

wsm = wrf_load_helper.WRFSimMetadata(
    start_datetime  = exp_beg_time,
    data_interval   = wrfout_data_interval,
    frames_per_file = args.frames_per_wrfout_file,
)

# =================================================================


def computeBudget(dt, output_filename):

    result = dict(status="UNKNOWN", dt=dt, output_filename=output_filename)

    try:
        # Loading data
        timestep_seconds = wrfout_data_interval / pd.Timedelta(seconds=1)
        timestep_timedelta = pd.Timedelta(seconds=timestep_seconds)
        time_rng = [ dt - timestep_timedelta, dt + timestep_timedelta]
        print("Start loading data.")
        
        ds = wrf_load_helper.loadWRFDataFromDir(
            wsm, 
            args.input_dir,
            beg_time = time_rng[0],
            end_time = time_rng[1],
            prefix="wrfout_d01_",
            avg=False,
            verbose=False,
            inclusive="both",
        )
        
        ds = ds.mean(dim=['south_north', 'south_north_stag'], keep_attrs=True)
      
        selected_time = [
            dt - timestep_timedelta
        ]

        Nx = ds.dims['west_east']
        Nz = ds.dims['bottom_top']
        
        dX   = ds.DX
        X_sU = dX * np.arange(Nx+1)
        X_sT = (X_sU[1:] + X_sU[:-1]) / 2
        X_T = np.repeat(np.reshape(X_sT, (1, -1)), [Nz,], axis=0)
        X_W = np.repeat(np.reshape(X_sT, (1, -1)), [Nz+1,], axis=0)
        
        ETA_W = ds.ZNW.to_numpy()[0, :]
        ETA_T = ds.ZNU.to_numpy()[0, :]

        Z_W = (ds.PHB + ds.PH).to_numpy() / 9.81
        Z_T = (Z_W[:, 1:, :] + Z_W[:, :-1, :]) / 2
        dZ_T = Z_W[:, 1:, :] - Z_W[:, :-1, :]
        dZ_W = np.zeros_like(Z_W)
        dZ_W[:, 1:-1, :] = (dZ_T[:, :-1, :] + dZ_T[:, 1:, :]) / 2.0
        dZ_W[:,  0, :] = dZ_W[:,  1, :]
        dZ_W[:, -1, :] = dZ_W[:, -2, :]
         
        # Create empty DataArray
        merge_data = []
        da_empty = xr.DataArray(
            data = np.zeros((1, Nz, Nx), dtype=np.float64),
            dims=["time", "z", "x"],
            coords=dict(
                X=(["x",], X_sT),
                ETA=(["z",], ETA_T),
                time=[dt,],
                reference_time=pd.Timestamp("2001-01-01"),
            ),
        )
     
        # Compute some common variables
        U_U = ds["U"].to_numpy()
        U_T = ( U_U[:, :, 1:] + U_U[:, :, :-1] ) / 2
     
        W_W = ds["W"].to_numpy()
        W_T = ( W_W[:, 1:, :] + W_W[:, :-1, :] ) / 2
     
        U_W = W_W.copy()
        U_W[:, 1:-1, :] = (U_T[:, :-1, :] + U_T[:, 1:, :]) / 2
        U_W[:, 0, :] = U_W[:, 1, :]
        U_W[:, -1, :] = U_W[:, -2, :]


        P_U = U_U.copy()
        P_U[:, :, 1:-1] = ( ds["P"][:, :, 1:] + ds["P"][:, :, :-1] ) / 2
        P_U[:, :, 0] = P_U[:, :, 1]
        P_U[:, :, -1] = P_U[:, :, -2]

        W_da = da_empty.copy().rename("W_T")
        W_da[:] = W_T[1, :, :]
        merge_data.append(W_da)


        P_T = (ds["P"] + ds["PB"]).to_numpy()
        THETA = (ds["T"] + T0).to_numpy()
        RHO = p0 / (Rd * THETA) * ( (P_T / p0)**(1 - Rd/cp) )

        # Compute du/dt
        dUdt = (U_T[2, :, :] - U_T[0, :, :]) / (2 * timestep_seconds)
        dUdt_da = da_empty.copy().rename("dUdt_T")
        dUdt_da[:] = dUdt
        merge_data.append(dUdt_da)

        # Compute dudx
        dUdX_T = (U_U[1, :, 1:] - U_U[1, :, :-1]) / dX
        dUdX_U = (U_U[1, :, 1:] - U_U[1, :, :-1]) / dX
        dUdX_da = da_empty.copy().rename("dUdX_T")
        dUdX_da[:] = dUdX_T
        merge_data.append(dUdX_da)

        # Compute Ududx
        dUdX_U = np.zeros_like(U_U)
        dUdX_U[:, :, 1:-1] = (U_U[:, :, 2:] - U_U[:, :, :-2]) / (2*dX)
        dUdX_U[:, :, 0]  = dUdX_U[:, :, 1]
        dUdX_U[:, :, -1] = dUdX_U[:, :, -2]

        # old way
        #UdUdX_T = U_T[1, :, :] * dUdX_T

        # new way
        UdUdX_U = U_U * dUdX_U
        UdUdX_T = (UdUdX_U[:, :, 1:] + UdUdX_U[:, :, :-1]) / 2
        UdUdX_da = da_empty.copy().rename("UdUdX_T")
        UdUdX_da[:] = UdUdX_T[1, :, :]
        merge_data.append(UdUdX_da)

        # Compute dudz
        dUdZ_T = (U_W[1, 1:, :] - U_W[1, :-1, :]) / dZ_T[1, :, :]
        dUdZ_da = da_empty.copy().rename("dUdZ_T")
        dUdZ_da[:] = dUdZ_T
        merge_data.append(dUdZ_da)

        # Compute wdudz
        dUdZ_T = (U_U[1, :, 1:] - U_U[1, :, :-1]) / dZ_T[1, :, :]
        WdUdZ_T = W_T[1, :, :] * dUdZ_T
        WdUdZ_da = da_empty.copy().rename("WdUdZ_T")
        WdUdZ_da[:] = WdUdZ_T
        merge_data.append(WdUdZ_da)

        # Compute fv
        f_T = ds["F"].to_numpy()  # time, X
        V_T = ds["V"].to_numpy()  # time, Z, X
        fV_T = f_T[:, None, :] * V_T

        fV_da = da_empty.copy().rename("fV_T")
        fV_da[:] = fV_T[1, :, :]
        merge_data.append(fV_da)

        # Compute dpdx
        dPdX_U = np.zeros_like(U_U)
        dPdX_U[:, :, 1:-1] = (P_T[:, :, 1:] - P_T[:, :, :-1]) / dX
        dPdX_U[:, :, 0]  = dPdX_U[:, :, 1]
        dPdX_U[:, :, -1] = dPdX_U[:, :, -2]
        dPdX_T = (dPdX_U[:, :, 1:] + dPdX_U[:, :, :-1]) / 2


        #dPdX_T = (P_U[1, :, 1:] - P_U[1, :, :-1]) / dX
        dPdX_da = da_empty.copy().rename("dPdX_T")
        dPdX_da[:] = dPdX_T[1, :, :]
        merge_data.append(dPdX_da)

        # Compute turbulent stress
        dUdZ_W = np.zeros_like(W_W)
        dUdZ_W[:, 1:-1, :] = U_T[:, 1:, :] - U_T[:, :-1, :]
        dUdZ_W /= dZ_W
        
        FLUX_W = dUdZ_W * ds["EXCH_M"].to_numpy()
        RNLDSTRS_T = ( FLUX_W[:, 1:, :] - FLUX_W[:, :-1, :] ) / dZ_T
        RNLDSTRS_da = da_empty.copy().rename("RNLDSTRS_T")
        RNLDSTRS_da[:] = RNLDSTRS_T[1, :, :]
        merge_data.append(RNLDSTRS_da)

        # Compute dw/dt
        W_W = ds["W"].to_numpy()
        W_T = ( W_W[:, 1:, :] + W_W[:, :-1, :] ) / 2
        dWdt = (W_T[2, :, :] - W_T[0, :, :]) / (2 * timestep_seconds)
        dWdt_da = da_empty.copy().rename("dWdt_T")
        dWdt_da[:] = dWdt
        merge_data.append(dWdt_da)


        # Compute dvdx

        # Compute dudz

        # Compute Umom_res
        Umom_res = dUdt_da + UdUdX_da + WdUdZ_da - fV_da + dPdX_da / RHO[1:2, :, :] - RNLDSTRS_da
        Umom_res = Umom_res.rename("Umom_res")
        merge_data.append(Umom_res)



        # ============= Thermal =============

        # Compute Ududx
        dTdX_U = np.zeros_like(U_U)
        dUdX_U[:, :, 1:-1] = (U_U[:, :, 2:] - U_U[:, :, :-2]) / (2*dX)
        dUdX_U[:, :, 0]  = dUdX_U[:, :, 1]
        dUdX_U[:, :, -1] = dUdX_U[:, :, -2]

        # old way
        #UdUdX_T = U_T[1, :, :] * dUdX_T

        # new way
        UdUdX_U = U_U * dUdX_U
        UdUdX_T = (UdUdX_U[:, :, 1:] + UdUdX_U[:, :, :-1]) / 2
        UdUdX_da = da_empty.copy().rename("UdUdX_T")
        UdUdX_da[:] = UdUdX_T[1, :, :]
        merge_data.append(UdUdX_da)

        # Compute dudz
        dUdZ_T = (U_W[1, 1:, :] - U_W[1, :-1, :]) / dZ_T[1, :, :]
        dUdZ_da = da_empty.copy().rename("dUdZ_T")
        dUdZ_da[:] = dUdZ_T
        merge_data.append(dUdZ_da)

        # Compute wdudz
        dUdZ_T = (U_U[1, :, 1:] - U_U[1, :, :-1]) / dZ_T[1, :, :]
        WdUdZ_T = W_T[1, :, :] * dUdZ_T
        WdUdZ_da = da_empty.copy().rename("WdUdZ_T")
        WdUdZ_da[:] = WdUdZ_T
        merge_data.append(WdUdZ_da)

















        ds = xr.merge(merge_data)
        print("Output file: %s" % (output_filename,))
        ds.to_netcdf(
            output_filename,
            unlimited_dims=["time",],
            encoding={
                'time': dict(
                    units = "seconds since %s" % exp_beg_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            },
        )

        result['status'] = 'OK'

    except Exception as e:
        
        result['status'] = 'ERROR'
        traceback.print_stack()
        print(e) 
    
    return result


dts = pd.date_range(time_beg, time_end, freq=wrfout_data_interval, inclusive="left")
failed_dates = []
input_args = []

for dt in dts:

    time_str = dt.strftime("%Y-%m-%d_%H:%M:%S")
    output_filename = os.path.join(args.output_dir, "wrf_budget_analysis_%s.nc" % time_str)

    if os.path.isfile(output_filename):
        print("[detect] File already exists for datetime =  %s." % (time_str, ))
    else:
        input_args.append((dt, output_filename,))
  
print("Create output dir: %s" % (args.output_dir,))
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

with Pool(processes=args.nproc) as pool:

    results = pool.starmap(computeBudget, input_args)

    for i, result in enumerate(results):
        if result['status'] != 'OK':
            print('!!! Failed to generate output of date %s.' % (result['dt'].strftime("%Y-%m-%d_%H"), ))

            failed_dates.append(result['dt'])


print("Tasks finished.")

print("Failed dates: ")
for i, failed_date in enumerate(failed_dates):
    print("%d : %s" % (i+1, failed_date.strftime("%Y-%m-%d %H:%M:%S"),))

print("Done.")



