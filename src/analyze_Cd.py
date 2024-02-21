import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import wrf_load_helper 
import datetime

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-dirs', type=str, nargs='+', help='Input directories.', required=True)
parser.add_argument('--output', type=str, help='Output filename in png.', default="")
parser.add_argument('--title', type=str, help='Title', default="")
parser.add_argument('--ref-exp-order', type=int, help='The reference case (start from 1) to perform decomposition', default=None)
parser.add_argument('--exp-names', type=str, nargs="+", help='Title', default=None)
parser.add_argument('--exp-beg-time', type=str, help='Title', required=True)
parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)

#parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in minutes after --exp-beg-time", required=True)
parser.add_argument('--x-rng', type=float, nargs=2, help="Spatial range to do the average", required=True)
parser.add_argument('--SST-rng', type=float, nargs=2, help='Title', default=[14.5, 16.5])
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--extra-title', type=str, help='Extra title', default="")
parser.add_argument('--time-format', type=str, help='Time format in the title. `hr` or `min`', default="hr", choices=["hr", "min"])

args = parser.parse_args()

print(args)

rho_a = 1.2     # kg / m^3
cp_a  = 1004.0  # J / kg / K
#C_T   = 1.5e-3  # scalar
C_T   = 7.5e-3  # scalar
Lq = 2.5e6
# [rho_a * cp_a * C_T] = J / m^3 / K
# [rho_a * cp_a * C_T] * m/s * K =  J / m^2 / s

if args.exp_names is None:
    args.exp_names = args.input_dirs
else:
    if len(args.input_dirs) != len(args.exp_names):
        raise Exception("Error: --exp-names does not receive the same length as --input-dirs")

exp_beg_time = pd.Timestamp(args.exp_beg_time)
time_beg = exp_beg_time + pd.Timedelta(minutes=args.time_rng[0])
time_end = exp_beg_time + pd.Timedelta(minutes=args.time_rng[1])

def relTimeInHrs(t):
    return (t - exp_beg_time.to_datetime64()) / np.timedelta64(1, 'h')

def relTimeInMins(t):
    return (t - exp_beg_time.to_datetime64()) / np.timedelta64(1, 'm')

if args.time_format == "hr":
    relTimeInFormat = relTimeInHrs
elif args.time_format == "min":
    relTimeInFormat = relTimeInMins
else:
    raise Exception("Unknown `--time-format`: %s" % (args.time_format,))


data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)

# Loading data
data = []
ref_ds = None
print("Start loading data.")
for i, input_dir in enumerate(args.input_dirs):
    print("Loading directory: %s" % (input_dir,))

    wsm = wrf_load_helper.WRFSimMetadata(
        start_datetime = exp_beg_time,
        data_interval = data_interval,
        frames_per_file = args.frames_per_wrfout_file,
    )

    ds = wrf_load_helper.loadWRFDataFromDir(
        wsm, 
        input_dir,
        beg_time = time_beg,
        end_time = time_end,
        prefix="wrfout_d01_",
        avg=None,
        verbose=False,
        inclusive="both",
    )


    ds = ds.mean(dim=['south_north', 'south_north_stag'], keep_attrs=True)
    if ref_ds is None:
    
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

    #WIND10 = ((ds["U10"]**2 + ds["V10"]**2)**0.5).rename("WIND10")
    
    dT = (np.amax(ds["TSK"].to_numpy()) - np.amin(ds["TSK"].to_numpy())) / 2 
    #WIND10 = ds["U10"].copy().rename("WIND10")
    #V10 = ds["V"].isel(bottom_top=0).to_numpy()
    #U10 = ds["U"].isel(bottom_top=0).to_numpy()
    #U10 = (U10[1:] + U10[:-1]) / 2.0

    #WIND10[:] = (U10**2 + V10**2)**0.5

    TOA    = (ds["TSK"] - ( 300.0 + ds["T"].isel(bottom_top=0))).rename("TOA")

    # Bolton (1980)
    E1 = 0.6112e3 * np.exp(17.67 * (ds["TSK"] - 273.15) / (ds["TSK"] - 29.65) )
    QSFCMR = 0.622 * E1 / (ds["PSFC"] - E1)
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


    theta_a = ds["T"].isel(bottom_top=0) + 300
    theta_a = theta_a.rename("theta_a") 
  
    Q = ds["QVAPOR"].isel(bottom_top=0) 
    Qvsh = Q / (1.0 + Q)
    Tvcon = 1 + 0.60772 * Qvsh
    theta_av = theta_a * Tvcon
    theta_av = theta_av.rename("theta_av")
 
    Linv_verify = 0.40 * 9.81 * ds["MOL"] / theta_av / ds["USTM"]**2
    Linv_verify = Linv_verify.rename("Linv_verify")

    merge_data.append(theta_a)
    merge_data.append(theta_a)
    merge_data.append(Linv_verify)
    
    ds = xr.merge(merge_data)
    
    ds = ds.where(
        (ds.coords["west_east"] >= args.x_rng[0]) & 
        (ds.coords["west_east"] <= args.x_rng[1]) 
    )
     
    ds.attrs["dT"]             = dT
    
    ds = ds.mean(dim=['time', 'west_east'], skipna=True, keep_attrs=True)
    data.append(ds)


print("Compute delta information...")

if args.ref_exp_order is not None:
    ref_exp_idx = args.ref_exp_order - 1
    ref_ds = data[ref_exp_idx]
else:
    raise Exception("Error: No ref_order specified.")    


# Weird offset of theta_*
Linv_offset = ref_ds["Linv_verify"].to_numpy() - ref_ds["RMOL"].to_numpy()
#for _data in data:
#    _data["RMOL"] += Linv_offset


merge_data = []
dTs = np.array([ _data.attrs["dT"] for _data in data])

for varname in ["dln_theta_a", "dln_UST", "dln_TST", "dln_Linv", "dln_Linv_verify", "Linv", "Linv_verify"]:
    
    empty_var = xr.DataArray(
        data=np.zeros_like(dTs),
        dims=["dT",],
        coords=dict(
            dT=(["dT",], dTs),
        ),
    ).rename(varname)
    
    merge_data.append(empty_var)

new_ds = xr.merge(merge_data)

for i, _data in enumerate(data):
   
    for var_ln, var_raw in [
        ["dln_UST", "UST"],
        ["dln_TST", "MOL"],
        ["dln_Linv", "RMOL"],
        ["dln_theta_a", "theta_a"],
    ]:
        _ref = ref_ds[var_raw].to_numpy()
        _delta = _data[var_raw].to_numpy() - _ref
        new_ds[var_ln][i] = _delta / _ref

    new_ds["Linv"][i] = _data["RMOL"].to_numpy()
    new_ds["Linv_verify"][i] = _data["Linv_verify"].to_numpy()

    #new_ds["Linv_verify"][i] = 0.40 * 9.81 * _data["MOL"] / (_data["theta_a"] * _data["UST"]**2)
 
  
new_ds["dln_Linv_verify"][:] = - 2 * new_ds["dln_UST"] - new_ds["dln_theta_a"] + new_ds["dln_TST"]
   

print("new_ds['dln_Linv'] = ", new_ds['dln_Linv'].to_numpy())


 
print("Done")

# =================================================================
print("Loading matplotlib...")
import matplotlib
if args.no_display:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
print("Done")


print("Plotting decomposition...")

fig, ax = plt.subplots(
    1,
    2,
    figsize=(12, 6),
    gridspec_kw=dict(hspace=0.3),
    sharex=True,
)


_ax = ax[0]

_ax.plot(new_ds.coords["dT"], new_ds["dln_Linv"], marker="o", color="black", label="$ \\Delta \\mathrm{ln} L^{-1}$")
_ax.plot(new_ds.coords["dT"], new_ds["dln_Linv_verify"], marker="o", color="black", linestyle="--", label="$ \\Delta \\mathrm{ln} L^{-1}_\\mathrm{veri}$")
_ax.plot(new_ds.coords["dT"], new_ds["dln_UST"], marker="x", label="$ \\Delta \\mathrm{ln} u_*$")
_ax.plot(new_ds.coords["dT"], new_ds["dln_TST"], marker="x", label="$ \\Delta \\mathrm{ln} \\theta_*$")
_ax.plot(new_ds.coords["dT"], new_ds["dln_theta_a"], marker="x", label="$ \\Delta \\mathrm{ln} \\theta_a$")

_ax.set_xlabel("$\\Delta T$ [K]")

_ax = ax[1]
_ax.plot(new_ds.coords["dT"], new_ds["Linv"], marker="o", color="black", label="$ \\mathrm{ln} L^{-1}$")
_ax.plot(new_ds.coords["dT"], new_ds["Linv_verify"], marker="o", color="black", linestyle="--", label="$ \\mathrm{ln} L^{-1}_\\mathrm{veri}$")

for _ax in ax.flatten():
    _ax.legend()
    _ax.grid(True)


#fig.suptitle("%sTime: %d ~ %d hr, $L^{-1}_\\mathrm{ref} = $ %.2e" % (args.extra_title, relTimeInHrs(time_beg), relTimeInHrs(time_end), ref_ds["RMOL"]))
fig.suptitle("%sTime: %d ~ %d %s, $L^{-1}_\\mathrm{ref} = $ %.2e" % (args.extra_title, relTimeInFormat(time_beg), relTimeInFormat(time_end), args.time_format, ref_ds["RMOL"]))
    

if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()



