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
    
    ds = xr.merge(merge_data)

    ds = ds.where(
        (ds.coords["west_east"] >= args.x_rng[0]) & 
        (ds.coords["west_east"] <= args.x_rng[1]) 
    )
    
    # Surface flux approximation comes from 
    # the file : module_sf_mynn.F    
    #Ulev1 = ds["U_T"].isel(bottom_top=0).to_numpy()
    #Vlev1 = ds["V"].isel(bottom_top=0).to_numpy()
    
    #WSPD1 = np.sqrt(Ulev1**2 + Vlev1**2)

    #TOA    = (ds["TSK"] - ds["T2"]).rename("TAO")

    HFX_approx = ds["FLHC"] * ds["TOA"]
    HFX_approx = HFX_approx.rename("HFX_approx")
    
    #y_full[j]    = C_T * (_ds["WIND10TAO"].to_numpy() - ref_ds["WIND10TAO"].to_numpy())
    #y_dWIND10[j] = C_T * ((_ds["WIND10"].to_numpy() - ref_ds["WIND10"].to_numpy()) * ref_ds["TAO"].to_numpy())
    #y_dTAO[j]    = C_T * (ref_ds["WIND10"].to_numpy() * (_ds["TAO"].to_numpy() - ref_ds["TAO"].to_numpy() )) 

    QFX_approx = ds["FLQC"] * ds["QOA"]
    _tmp = QFX_approx.to_numpy()
    _tmp[_tmp <= -0.02] = -0.02
    QFX_approx[:] = _tmp
    LH_approx = Lq * QFX_approx
    LH_approx = LH_approx.rename("LH_approx")
   
    ds = xr.merge([ds, HFX_approx, LH_approx])

    FLHC_mean = ds["FLHC"].mean(dim="west_east").rename("FLHC_mean")
    TOA_mean = ds["TOA"].mean(dim="west_east").rename("TOA_mean")
    FLHCp = (ds["FLHC"] - FLHC_mean).rename("FLHCp")
    TOAp  = (ds["TOA"]  - TOA_mean).rename("TOAp")
    FLHCpTOAp_mean = (FLHCp * TOAp).mean(dim="west_east").rename("FLHCpTOAp_mean")

    FLQC_mean = ds["FLQC"].mean(dim="west_east").rename("FLQC_mean")
    QOA_mean = ds["QOA"].mean(dim="west_east").rename("QOA_mean")
    FLQCp = (ds["FLQC"] - FLQC_mean).rename("FLQCp")
    QOAp  = (ds["QOA"]  - QOA_mean).rename("QOAp")
    FLQCpQOAp_mean = (FLQCp * QOAp).mean(dim="west_east").rename("FLQCpQOAp_mean")
    
    ds = xr.merge([ds, 

        FLHC_mean, TOA_mean,
        FLHCp, TOAp,
        FLHCpTOAp_mean,

        FLQC_mean, QOA_mean,
        FLQCp, QOAp,
        FLQCpQOAp_mean,

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

    y_HFX_full    = np.zeros_like(x)
    y_HFX_dFLHC   = np.zeros_like(x)
    y_HFX_dTOA    = np.zeros_like(x)
    y_HFX_dcorr   = np.zeros_like(x)
    y_HFX_res     = np.zeros_like(x)
    y_HFX_ref     = ref_ds["HFX"].to_numpy()
 
    y_LH_full    = np.zeros_like(x)
    y_LH_dFLQC   = np.zeros_like(x)
    y_LH_dQOA    = np.zeros_like(x)
    y_LH_dcorr   = np.zeros_like(x)
    y_LH_res     = np.zeros_like(x)
    y_LH_ref     = ref_ds["LH"].to_numpy()
    
    y_HFX = np.zeros_like(x)
    y_LH  = np.zeros_like(x)


    print("Doing decomposition")
    for j, input_dir in enumerate(args.input_dirs):
       
        print("j => %d" % (j,)) 
        _ds = data[j]

        print(_ds)
        print(ref_ds)

        y_HFX[j] = _ds["HFX"] - ref_ds["HFX"]
        y_LH[j] = _ds["LH"] - ref_ds["LH"]

        y_HFX_full[j]     = _ds["HFX_approx"] - ref_ds["HFX_approx"]
        y_HFX_dFLHC[j]    = ( _ds["FLHC_mean"] - ref_ds["FLHC_mean"] ) * ref_ds["TOA_mean"]
        y_HFX_dTOA[j]     = ( _ds["TOA_mean"] - ref_ds["TOA_mean"] ) * ref_ds["FLHC_mean"]
        y_HFX_dcorr[j]    = ( _ds["FLHCpTOAp_mean"] - ref_ds["FLHCpTOAp_mean"] )

        y_LH_full[j]     = _ds["LH_approx"] - ref_ds["LH_approx"]
        y_LH_dFLQC[j]    = Lq * ( _ds["FLQC_mean"] - ref_ds["FLQC_mean"] ) * ref_ds["QOA_mean"]
        y_LH_dQOA[j]     = Lq * ( _ds["QOA_mean"] - ref_ds["QOA_mean"] ) * ref_ds["FLQC_mean"]
        y_LH_dcorr[j]    = Lq * ( _ds["FLQCpQOAp_mean"] - ref_ds["FLQCpQOAp_mean"] )

        x[j] = _ds.attrs["dT"]
        
        
    y_HFX_res = y_HFX_full - ( y_HFX_dFLHC + y_HFX_dTOA + y_HFX_dcorr )
    y_LH_res  = y_LH_full  - ( y_LH_dFLQC  + y_LH_dQOA  + y_LH_dcorr  )

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
    ["Full", y_HFX_full,   ("black", "-")],
    ["Wind", y_HFX_dFLHC,  ("dodgerblue", "-")],
    ["Temp", y_HFX_dTOA,   ("red"       , "-")],
    ["Corr", y_HFX_dcorr,  ("magenta"   , "-")],
    ["Res",  y_HFX_res,    ("green"     , "--")],
    ["HFX",  y_HFX,        ("gray"     , "--")],
])
plot_bundles.append(dict(title="Sensible heat", plotting_list=plotting_list, ref_val=y_HFX_ref))


plotting_list = []
plotting_list.extend([
    ["Full", y_LH_full,   ("black", "-")],
    ["Wind", y_LH_dFLQC,  ("dodgerblue", "-")],
    ["Vapor", y_LH_dQOA,   ("red"       , "-")],
    ["Corr", y_LH_dcorr,  ("magenta"   , "-")],
    ["Res",  y_LH_res,    ("green"     , "--")],
    ["LH",  y_LH,         ("gray"     , "--")],
])
plot_bundles.append(dict(title="Latent heat", plotting_list=plotting_list, ref_val=y_LH_ref))

fig, ax = plt.subplots(len(plot_bundles), 1, figsize=(4, 6), sharex=True)

for i, plot_bundle in enumerate(plot_bundles):

    _ax = ax[i]
    for decomp_name, _y, (color, ls) in plot_bundle["plotting_list"]:
        _ax.scatter(x, _y, s=20, c=color)
        _ax.plot(x, _y, color=color, linestyle=ls, label=decomp_name)
        
        
    _ax.set_title("%s (ref value = %.1f)" % (plot_bundle["title"], plot_bundle["ref_val"],))
    _ax.set_ylabel("[$\\mathrm{W} / \\mathrm{m}^2$]")
    _ax.legend(loc="upper left", ncol=2)
    _ax.grid(True)

time_fmt = "%Y/%m/%d %Hh"
fig.suptitle("Time: %s ~ %s\nAverage: %d ~ %d km" % (time_beg.strftime(time_fmt), time_end.strftime(time_fmt), args.x_rng[0], args.x_rng[1]))

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



