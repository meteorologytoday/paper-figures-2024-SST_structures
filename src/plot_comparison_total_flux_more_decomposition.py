import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import diagnostics
import wrf_load_helper 
import datetime

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-dirs', type=str, nargs='+', help='Input directories.', required=True)
parser.add_argument('--output', type=str, help='Output filename in png.', default="")
parser.add_argument('--output-decomp', type=str, help='Output filename in png.', default="")
parser.add_argument('--title', type=str, help='Title', default="")
parser.add_argument('--ref-exp-order', type=int, help='The reference case (start from 1) to perform decomposition', default=None)
parser.add_argument('--exp-names', type=str, nargs="+", help='Title', default=None)
parser.add_argument('--exp-beg-time', type=str, help='Title', required=True)
parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)

parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
parser.add_argument('--x-rng', type=float, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
parser.add_argument('--LH-rng', type=float, nargs=2, help="LH range", default=[20, 100])
parser.add_argument('--HFX-rng', type=float, nargs=2, help="HFX range", default=[-20, 40])
parser.add_argument('--blh-method', type=str, help='Method to determine boundary layer height', default=["grad", "bulk"], nargs='+', choices=['bulk', 'grad'])
parser.add_argument('--SST-rng', type=float, nargs=2, help='Title', default=[14.5, 16.5])
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--plot-check', action="store_true")

parser.add_argument('--plot-HFX', action="store_true", help='If to plot HFX in the figure')

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
        avg=False,
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
    
    #y_full[j]    = C_T * (_ds["WIND10TAO"].to_numpy() - ref_ds["WIND10TAO"].to_numpy())
    #y_dWIND10[j] = C_T * ((_ds["WIND10"].to_numpy() - ref_ds["WIND10"].to_numpy()) * ref_ds["TAO"].to_numpy())
    #y_dTAO[j]    = C_T * (ref_ds["WIND10"].to_numpy() * (_ds["TAO"].to_numpy() - ref_ds["TAO"].to_numpy() )) 

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

    C_Q_m, C_Q_p = horDecomp(ds["C_Q"], "C_Q_m", "C_Q_p")
    QOA_m, QOA_p = horDecomp(ds["QOA"], "QOA_m", "QOA_p")

    C_Q_QOA_cx = (C_Q_p * QOA_p).mean(dim="west_east").rename("C_Q_QOA_cx")
    C_Q_WND_cx = (C_Q_p * WND_p).mean(dim="west_east").rename("C_Q_WND_cx")
    WND_QOA_cx = (WND_p * QOA_p).mean(dim="west_east").rename("WND_QOA_cx")
    C_Q_WND_QOA_cx = (C_Q_p * WND_p * QOA_p).mean(dim="west_east").rename("C_Q_WND_QOA_cx")


    ds = xr.merge([ds, 

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

    ])

    ds.attrs["dT"]             = dT

    ds = ds.mean(dim=['time', 'west_east'], skipna=True, keep_attrs=True)
    data.append(ds)

print("Done")



# =================================================================
# Figure: HFX decomposition
# =================================================================

if args.ref_exp_order is not None:
    print("Analyze the contribution to HFX")

    # Finding reference exp
    ref_exp_idx = args.ref_exp_order - 1
    
    ref_ds = data[ref_exp_idx]

    x = np.zeros((len(args.input_dirs),))

    y_HFX = { k : np.zeros_like(x) for k in [
        "real", "full", "d(C_H)_WND_TOA", "C_H_dWND_TOA", "C_H_WND_dTOA",
        "dWND_C_H_TOA_cx", "WND_d(C_H_TOA_cx)",
        "dC_H_WND_TOA_cx", "C_H_d(WND_TOA_cx)",
        "dTOA_C_H_WND_cx", "TOA_d(C_H_WND_cx)",
        "d(C_H_WND_TOA_cx)",
    ]}

    y_LH = { k : np.zeros_like(x) for k in [
        "real", "full", "d(C_Q)_WND_QOA", "C_Q_dWND_QOA", "C_Q_WND_dQOA",
        "dWND_C_Q_QOA_cx", "WND_d(C_Q_QOA_cx)",
        "dC_Q_WND_QOA_cx", "C_Q_d(WND_QOA_cx)",
        "dQOA_C_Q_WND_cx", "QOA_d(C_Q_WND_cx)",
        "d(C_Q_WND_QOA_cx)",
    ]}

    y_HFX["ref"] = ref_ds["HFX"].to_numpy()
    y_LH["ref"]  = ref_ds["LH"].to_numpy()

    ref_WND_m = ref_ds["WND_m"]
    ref_C_H_m = ref_ds["C_H_m"]
    ref_C_Q_m = ref_ds["C_Q_m"]
    ref_TOA_m = ref_ds["TOA_m"]
    ref_QOA_m = ref_ds["QOA_m"]
 
    print("Doing decomposition")
    for j, input_dir in enumerate(args.input_dirs):
       
        print("j => %d" % (j,)) 
        _ds = data[j]

        y_HFX["real"][j] = _ds["HFX"] - ref_ds["HFX"]

        y_HFX["full"][j]     = _ds["HFX_approx"] - ref_ds["HFX_approx"]
        y_HFX["d(C_H)_WND_TOA"][j]  = ( _ds["C_H_m"] - ref_ds["C_H_m"] ) * ref_ds["WND_m"] * ref_ds["TOA_m"]
        y_HFX["C_H_dWND_TOA"][j]    = ref_ds["C_H_m"] * ( _ds["WND_m"] - ref_ds["WND_m"] ) * ref_ds["TOA_m"]
        y_HFX["C_H_WND_dTOA"][j]   = ref_ds["C_H_m"] * ref_ds["WND_m"] * ( _ds["TOA_m"] - ref_ds["TOA_m"] )
        
        y_HFX["dWND_C_H_TOA_cx"][j]   = ( _ds["WND_m"] - ref_ds["WND_m"] ) * ref_ds["C_H_TOA_cx"]
        y_HFX["WND_d(C_H_TOA_cx)"][j] = ref_ds["WND_m"] * ( _ds["C_H_TOA_cx"] - ref_ds["C_H_TOA_cx"] )

        y_HFX["dC_H_WND_TOA_cx"][j]   = ( _ds["C_H_m"] - ref_ds["C_H_m"] ) * ref_ds["WND_TOA_cx"]
        y_HFX["C_H_d(WND_TOA_cx)"][j] = ref_ds["C_H_m"] * ( _ds["WND_TOA_cx"] - ref_ds["WND_TOA_cx"] )

        y_HFX["dTOA_C_H_WND_cx"][j]   = ( _ds["TOA_m"] - ref_ds["TOA_m"] ) * ref_ds["C_H_WND_cx"]
        y_HFX["TOA_d(C_H_WND_cx)"][j] = ref_ds["TOA_m"] * ( _ds["C_H_WND_cx"] - ref_ds["C_H_WND_cx"] )
        
        y_HFX["d(C_H_WND_TOA_cx)"][j] = ( _ds["C_H_WND_TOA_cx"] - ref_ds["C_H_WND_TOA_cx"] )



        y_LH["real"][j]  = _ds["LH"] - ref_ds["LH"]

        y_LH["full"][j]     = _ds["LH_approx"] - ref_ds["LH_approx"]
        y_LH["d(C_Q)_WND_QOA"][j]  = Lq * ( _ds["C_Q_m"] - ref_ds["C_Q_m"] ) * ref_ds["WND_m"] * ref_ds["QOA_m"]
        y_LH["C_Q_dWND_QOA"][j]    = Lq * ref_ds["C_Q_m"] * ( _ds["WND_m"] - ref_ds["WND_m"] ) * ref_ds["QOA_m"]
        y_LH["C_Q_WND_dQOA"][j]   = Lq * ref_ds["C_Q_m"] * ref_ds["WND_m"] * ( _ds["QOA_m"] - ref_ds["QOA_m"] )
        
        y_LH["dWND_C_Q_QOA_cx"][j]   = Lq * ( _ds["WND_m"] - ref_ds["WND_m"] ) * ref_ds["C_Q_QOA_cx"]
        y_LH["WND_d(C_Q_QOA_cx)"][j] = Lq * ref_ds["WND_m"] * ( _ds["C_Q_QOA_cx"] - ref_ds["C_Q_QOA_cx"] )

        y_LH["dC_Q_WND_QOA_cx"][j]   = Lq * ( _ds["C_Q_m"] - ref_ds["C_Q_m"] ) * ref_ds["WND_QOA_cx"]
        y_LH["C_Q_d(WND_QOA_cx)"][j] = Lq * ref_ds["C_Q_m"] * ( _ds["WND_QOA_cx"] - ref_ds["WND_QOA_cx"] )

        y_LH["dQOA_C_Q_WND_cx"][j]   = Lq * ( _ds["QOA_m"] - ref_ds["QOA_m"] ) * ref_ds["C_Q_WND_cx"]
        y_LH["QOA_d(C_Q_WND_cx)"][j] = Lq * ref_ds["QOA_m"] * ( _ds["C_Q_WND_cx"] - ref_ds["C_Q_WND_cx"] )
        
        y_LH["d(C_Q_WND_QOA_cx)"][j] = Lq * ( _ds["C_Q_WND_QOA_cx"] - ref_ds["C_Q_WND_QOA_cx"] )


        x[j] = _ds.attrs["dT"]
        
        
    y_HFX["res"] = y_HFX["full"] - (
            y_HFX["d(C_H)_WND_TOA"]
        +   y_HFX["C_H_dWND_TOA"]
        +   y_HFX["C_H_WND_dTOA"]
        +   y_HFX["dWND_C_H_TOA_cx"]
        +   y_HFX["WND_d(C_H_TOA_cx)"]
        +   y_HFX["dC_H_WND_TOA_cx"]
        +   y_HFX["C_H_d(WND_TOA_cx)"]
        +   y_HFX["dTOA_C_H_WND_cx"]
        +   y_HFX["TOA_d(C_H_WND_cx)"]
        +   y_HFX["d(C_H_WND_TOA_cx)"]
    )

    y_LH["res"] = y_LH["full"] - (
            y_LH["d(C_Q)_WND_QOA"]
        +   y_LH["C_Q_dWND_QOA"]
        +   y_LH["C_Q_WND_dQOA"]
        +   y_LH["dWND_C_Q_QOA_cx"]
        +   y_LH["WND_d(C_Q_QOA_cx)"]
        +   y_LH["dC_Q_WND_QOA_cx"]
        +   y_LH["C_Q_d(WND_QOA_cx)"]
        +   y_LH["dQOA_C_Q_WND_cx"]
        +   y_LH["QOA_d(C_Q_WND_cx)"]
        +   y_LH["d(C_Q_WND_QOA_cx)"]
    )



else:

    raise Exception("Error: No ref_order specified.")    



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
plot_bundles = []

plotting_list = []
plotting_list.extend([
    ["Full", y_HFX["full"],   ("black", "--")],
    ["$ \\Delta \\left(\\overline{C}\\right)_H \\overline{U} \\overline{T}_{OA} $", y_HFX["d(C_H)_WND_TOA"],  ("dodgerblue", "-")],
    ["$ \\overline{C}_H \\Delta \\left(\\overline{U}\\right) \\overline{T}_{OA} $", y_HFX["C_H_dWND_TOA"],  ("dodgerblue", "-")],
    ["$ \\overline{C}_H \\overline{U} \\Delta \\left(\\overline{T}\\right)_{OA} $", y_HFX["C_H_WND_dTOA"],  ("dodgerblue", "-")],

    ["$ \\Delta \\overline{U}  \\overline{ C'_H T'_{OA} }$", y_HFX["dWND_C_H_TOA_cx"],  ("dodgerblue", "-")],
    ["$ \\overline{U}  \\Delta \\left( \\overline{ C'_H T'_{OA} }\\right) $", y_HFX["WND_d(C_H_TOA_cx)"],  ("dodgerblue", "-")],

    ["$ \\Delta \\overline{C}_H  \\overline{ U' T'_{OA} }$", y_HFX["dC_H_WND_TOA_cx"],  ("dodgerblue", "-")],
    ["$ \\overline{C}_H  \\Delta \\left( \\overline{ U' T'_{OA}}\\right) $", y_HFX["C_H_d(WND_TOA_cx)"],  ("dodgerblue", "-")],

    ["$ \\Delta \\overline{T}_{OA}  \\overline{ C'_H U' }$", y_HFX["dTOA_C_H_WND_cx"],  ("dodgerblue", "-")],
    ["$ \\overline{T}_{OA}  \\Delta \\left( \\overline{ C'_H  U'} \\right) $", y_HFX["TOA_d(C_H_WND_cx)"],  ("dodgerblue", "-")],
    ["$ \\Delta \\left( \\overline{ C'_H  U' T'_{OA} }\\right) $", y_HFX["d(C_H_WND_TOA_cx)"],  ("dodgerblue", "-")],
    ["Res", y_HFX["res"],  ("gray", "--")],

])
plot_bundles.append(dict(title="Sensible heat", plotting_list=plotting_list, ref_val=y_HFX["ref"]))


plotting_list = []

plotting_list.extend([
    ["Full", y_LH["full"],   ("black", "--")],
    ["$ \\Delta \\left(\\overline{C}\\right)_Q \\overline{U} \\overline{Q}_{OA} $", y_LH["d(C_Q)_WND_QOA"],  ("dodgerblue", "-")],
    ["$ \\overline{C}_Q \\Delta \\left(\\overline{U}\\right) \\overline{Q}_{OA} $", y_LH["C_Q_dWND_QOA"],  ("dodgerblue", "-")],
    ["$ \\overline{C}_Q \\overline{U} \\Delta \\left(\\overline{Q}\\right)_{OA} $", y_LH["C_Q_WND_dQOA"],  ("dodgerblue", "-")],

    ["$ \\Delta \\overline{U}  \\overline{ C'_Q Q'_{OA} }$", y_LH["dWND_C_Q_QOA_cx"],  ("dodgerblue", "-")],
    ["$ \\overline{U}  \\Delta \\left( \\overline{ C'_Q Q'_{OA} }\\right) $", y_LH["WND_d(C_Q_QOA_cx)"],  ("dodgerblue", "-")],

    ["$ \\Delta \\overline{C}_Q  \\overline{ U' Q'_{OA} }$", y_LH["dC_Q_WND_QOA_cx"],  ("dodgerblue", "-")],
    ["$ \\overline{C}_Q  \\Delta \\left( \\overline{ U' Q'_{OA}}\\right) $", y_LH["C_Q_d(WND_QOA_cx)"],  ("dodgerblue", "-")],

    ["$ \\Delta \\overline{Q}_{OA}  \\overline{ C'_Q U' }$", y_LH["dQOA_C_Q_WND_cx"],  ("dodgerblue", "-")],
    ["$ \\overline{Q}_{OA}  \\Delta \\left( \\overline{ C'_Q  U'} \\right) $", y_LH["QOA_d(C_Q_WND_cx)"],  ("dodgerblue", "-")],
    ["$ \\Delta \\left( \\overline{ C'_Q  U' Q'_{OA} }\\right) $", y_LH["d(C_Q_WND_QOA_cx)"],  ("dodgerblue", "-")],
    ["Res", y_LH["res"],  ("gray", "--")],

])

plot_bundles.append(dict(title="Latent heat", plotting_list=plotting_list, ref_val=y_LH["ref"]))

fig, ax = plt.subplots(len(plot_bundles), 1, figsize=(8, 12), sharex=True)

for i, plot_bundle in enumerate(plot_bundles):

    _ax = ax[i]
    for decomp_name, _y, (color, ls) in plot_bundle["plotting_list"]:
        #_ax.scatter(x, _y, s=20, c=color)
        _ax.scatter(x, _y, s=20)
        #_ax.plot(x, _y, color=color, linestyle=ls, label=decomp_name)
        _ax.plot(x, _y, linestyle=ls, label=decomp_name)
        
        
    _ax.set_title("%s (ref value = %.1f)" % (plot_bundle["title"], plot_bundle["ref_val"],))
    _ax.set_ylabel("[$\\mathrm{W} / \\mathrm{m}^2$]")
    _ax.legend(loc="upper left", ncol=2)
    _ax.grid(True)

time_fmt = "%Y/%m/%d %Hh"
fig.suptitle("Time: %s ~ %s\nAverage: %d ~ %d km\n($\\overline{C}_H = %.1f$, $L_q \\overline{C}_Q = %.1f$, $\\overline{U} = %.1f$, $\\overline{T}_{OA} = %.1f \\mathrm{K}$, $\\overline{Q}_{OA} = %.1f \\mathrm{g}/\\mathrm{kg}$)" % (
    time_beg.strftime(time_fmt),
    time_end.strftime(time_fmt),
    args.x_rng[0],
    args.x_rng[1],
    ref_C_H_m,
    ref_C_Q_m * Lq,
    ref_WND_m,
    ref_TOA_m,
    ref_QOA_m * 1e3,
))

ax[-1].set_xlabel("Amplitude [ K ]")

if args.output_decomp != "":
    print("Saving output: ", args.output_decomp)
    fig.savefig(args.output_decomp, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()


plot_infos = dict(

    LH = dict(
        rng = args.LH_rng,
        #ticks = np.arange(-40, 61, 20),
        label = "LH",
        unit = "$\\mathrm{W} / \\mathrm{m}^2$",
    ),

    LH_approx = dict(
        rng = args.LH_rng,
        #ticks = np.arange(-40, 61, 20),
        label = "LH approx",
        unit = "$\\mathrm{W} / \\mathrm{m}^2$",
    ),


    HFX = dict(
        rng = args.HFX_rng,
        #ticks = np.arange(-40, 61, 20),
        label = "HFX",
        unit = "$\\mathrm{W} / \\mathrm{m}^2$",
    ),

    HFX_approx = dict(
        rng = args.HFX_rng,
        #ticks = np.arange(-40, 61, 20),
        label = "HFX approx",
        unit = "$\\mathrm{W} / \\mathrm{m}^2$",
    ),


    TSK = dict(
        offset = 273.15,
        rng = [13.5, 16.5],
        label = "TSK",
        unit = "${}^\\circ\\mathrm{C}$",
    ),

    U10 = dict(
        rng = [3, 10],
        unit = "$\\mathrm{m} / \\mathrm{s}$",
    ),

    V10 = dict(
        rng = [0, 5],
        unit = "$\\mathrm{m} / \\mathrm{s}$",
    ),

)

# =================================================================
# Figure: Variable as a function of x
# =================================================================

plot_varnames = [ ["HFX", "HFX_approx"], ["LH", "LH_approx"]]

fig, ax = plt.subplots(len(plot_varnames), 1, figsize=(8, 6), squeeze=False, sharex=True)


for i, varnames in enumerate(plot_varnames):

    print("Plotting variables: %s" % (varnames,))        
    _ax = ax[i, 0]

    for k, varname in enumerate(varnames):
        plot_info = plot_infos[varname]

        x = []
        y = []
        for j, input_dir in enumerate(args.input_dirs):
            
            _ds = data[j]
            _d = _ds[varname].to_numpy()
            if "offset" in plot_info:
                _d -= plot_info["offset"]
           
            x.append(_ds.attrs["dT"])
            y.append(_d)
           
        c = ["black", "red", "green"][k] 
        _ax.scatter(x, y, s=20, c=c)
        _ax.plot(x, y, color=c)
   
    
    plot_info = plot_infos[varnames[0]]
    
 
    if "rng" in plot_info:
        pass
        #_ax.set_ylim(plot_info["rng"])
    
    if "ticks" in plot_info:
        _ax.set_yticks(plot_info["ticks"])
    
    if "label" in plot_info:
        _ax.set_title(plot_info["label"])
    else:
        _ax.set_title(varname)

    if "unit" in plot_info:
        _ax.set_ylabel("[%s]" % (plot_info["unit"],))

    #_ax.legend()
    _ax.grid(True)

    
    #ax[0,0].set_title("Time: %s ~ %s" % (ref_ds.time[0].strftime("%H:%M"), ref_ds.time[-1].strftime("%H:%M")))

ax.flatten()[-1].set_xlabel("Amplitude [ K ]")

#ax[0,0].set_xlim([0,1500])
time_fmt = "%Y/%m/%d %Hh"
fig.suptitle("Time: %s ~ %s\nAverage: %d ~ %d km" % (time_beg.strftime(time_fmt), time_end.strftime(time_fmt), args.x_rng[0], args.x_rng[1]))



if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()



