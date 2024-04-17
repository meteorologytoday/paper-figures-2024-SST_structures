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

parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
parser.add_argument('--x-rng', type=float, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
parser.add_argument('--SST-rng', type=float, nargs=2, help='Title', default=[14.5, 16.5])
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--plot-check', action="store_true")
parser.add_argument('--extra-title', type=str, help='Extra title', default="")

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
time_beg = exp_beg_time + pd.Timedelta(hours=args.time_rng[0])
time_end = exp_beg_time + pd.Timedelta(hours=args.time_rng[1])

def relTimeInHrs(t):
    return (t - exp_beg_time.to_datetime64()) / np.timedelta64(1, 'h')


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
   
    ds = xr.merge(merge_data)

    ds = ds.where(
        (ds.coords["west_east"] >= args.x_rng[0]) & 
        (ds.coords["west_east"] <= args.x_rng[1]) 
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
    
    def horDecomp(da, name_m="mean", name_p="prime"):
        m = da.mean(dim="west_east").rename(name_m)
        #p = (da - m).rename(name_p) 
        p = da.rename(name_p) 
        return m, p

    WND_m, WND_p = horDecomp(ds["WND_sfc"], "WND_m", "WND_p")
    
    C_H_m, C_H_p = horDecomp(ds["C_H"], "C_H_m", "C_H_p")
    TOA_m, TOA_p = horDecomp(ds["TOA"], "TOA_m", "TOA_p")
    
    C_H_TOA_cx = (C_H_p * TOA_p).mean(dim="west_east").rename("C_H_TOA_cx")
    C_H_WND_cx = (C_H_p * WND_p).mean(dim="west_east").rename("C_H_WND_cx")
    WND_TOA_cx = (WND_p * TOA_p).mean(dim="west_east").rename("WND_TOA_cx")
    C_H_WND_TOA_cx = (C_H_p * WND_p * TOA_p).mean(dim="west_east").rename("C_H_WND_TOA_cx")

    C_Q_m, C_Q_p = horDecomp(ds["C_Q"], "C_Q_m", "C_Q_p")
    QOA_m, QOA_p = horDecomp(ds["QOA"], "QOA_m", "QOA_p")

    C_Q_QOA_cx = (C_Q_p * QOA_p).mean(dim="west_east").rename("C_Q_QOA_cx")
    C_Q_WND_cx = (C_Q_p * WND_p).mean(dim="west_east").rename("C_Q_WND_cx")
    WND_QOA_cx = (WND_p * QOA_p).mean(dim="west_east").rename("WND_QOA_cx")
    C_Q_WND_QOA_cx = (C_Q_p * WND_p * QOA_p).mean(dim="west_east").rename("C_Q_WND_QOA_cx")
    
    SST_m, SST_p = horDecomp(ds["TSK"], "SST_m", "SST_p")

    LST = ( ds["RMOL"] ** (-1) ).rename("LST")

    ds = xr.merge([ds, 

        SST_m, SST_p,
        WND_m, WND_p,

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

        LST,
    ])

    ds.attrs["dT"]             = dT

    ds = ds.mean(dim=['time',], skipna=True, keep_attrs=True)
    data.append(ds)

print("Done")



# =================================================================
# Figure: HFX decomposition
# =================================================================

if args.ref_exp_order is not None:
    ref_exp_idx = args.ref_exp_order - 1
    ref_ds = data[ref_exp_idx]
else:
    raise Exception("Error: No ref_order specified.")    


plot_infos = dict(

    WND_p = dict(
        label = "$ \\left| \\vec{U}_\\mathrm{sfc} \\right|'$",
    ),

    TOA_p = dict(
        label = "$ T_\\mathrm{OA}' $",
    ),

    QOA_p = dict(
        label = "$ q_\\mathrm{OA}' $",
    ),

    C_H_p = dict(
        label = "$ C_H' $",
    ),

    C_Q_p = dict(
        label = "$ C_q' $",
    ),

    SST_p = dict(
        label = "$ \\mathrm{SST}' $",
    ),

    LST = dict(
        label = "$ L_* $",
    ),

    RMOL = dict(
        label = "$ L_*^{-1} $",
    ),

    MOL = dict(
        label = "$ \\theta_* $",
    ),

    UST = dict(
        label = "$ u_* $",
    ),

)

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
plot_vars = [
    "SST_p",
    "WND_p",
    "TOA_p",
    "QOA_p",
    "C_H_p",
    "C_Q_p",
    "UST",
    "MOL",
    "RMOL",
]

fig, ax = plt.subplots(
    len(plot_vars),
    1,
    figsize=(8, 12),
    gridspec_kw=dict(hspace=0.3),
    sharex=True,
)

color_starting = 0.8

for i, plot_var in enumerate(plot_vars):

    _ax = ax[i]

    plot_info = plot_infos[plot_var]

    for j in range(len(args.input_dirs)):
        _ds = data[j]

        color = tuple([ color_starting * ( 1.0 - j / len(args.input_dirs) ) ] * 3)

        #_ax.plot(X_sT, _ds[plot_var] - ref_ds[plot_var], color=color, label="%d" % (j,))
        _ax.plot(X_sT, _ds[plot_var], color=color, label="%d" % (j,))

    _ax.set_title("(%s) %s" % ("abcdefghijklmn"[i], plot_info["label"],))
    #_ax.set_ylabel("[$\\mathrm{W} / \\mathrm{m}^2$]")
    #_ax.legend(loc="upper right", ncol=5)
    _ax.grid(True)

ax[-1].set_xlabel("[km]")


#time_fmt = "%Y/%m/%d %Hh"
fig.suptitle("%sTime: %d ~ %d hr" % (args.extra_title, relTimeInHrs(time_beg), relTimeInHrs(time_end),))
    


#ax[-1].set_xlabel("Amplitude [ K ]")

if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()



