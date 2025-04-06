import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import datetime
import wrf_load_helper
import wrf_preprocess 
import cmocean
from shared_constants import *

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-dir', type=str, help='Input directory.', required=True)
parser.add_argument('--output', type=str, help='Output filename in png.', default="")
parser.add_argument('--extra-title', type=str, help='Title', default="")
parser.add_argument('--overwrite-title', type=str, help='If set then title will be set to this.', default="")

parser.add_argument('--no-display', action="store_true")
parser.add_argument('--plot-check', action="store_true")
parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
#parser.add_argument('--ref-time-rng', type=int, nargs=2, help="Reference time range in hours after --exp-beg-time", required=True)
parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
parser.add_argument('--z-rng', type=float, nargs=2, help='The plotted height rng in meters.', default=[0, 2000.0])
parser.add_argument('--THETA-rng', type=float, nargs=2, help='Theta range in K', default=[None, None])
parser.add_argument('--Nfreq2-rng', type=float, nargs=2, help='Theta range in 1e-2/s^2', default=[None, None])
parser.add_argument('--TKE-rng', type=float, nargs=2, help='The plotted surface wind in m/s', default=[None, None])
parser.add_argument('--DTKE-rng', type=float, nargs=2, help='The plotted surface wind in m/s', default=[None, None])
parser.add_argument('--U-rng', type=float, nargs=2, help='U range in m/s', default=[None, None])
parser.add_argument('--Q-rng', type=float, nargs=2, help='Q range in g/kg', default=[None, None])

parser.add_argument('--tke-analysis', type=str, help='analysis beg time', choices=["TRUE", "FALSE"], default="FALSE")
parser.add_argument('--thumbnail-skip', type=int, help='Skip of thumbnail numbering.', default=0)
parser.add_argument('--thumbnail-numbering', type=str, help='Skip of thumbnail numbering.', default="abcdefghijklmn")


args = parser.parse_args()

print(args)

    


exp_beg_time = pd.Timestamp(args.exp_beg_time)
wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)

time_beg = exp_beg_time + pd.Timedelta(minutes=args.time_rng[0])
time_end = exp_beg_time + pd.Timedelta(minutes=args.time_rng[1])

def relTimeInHrs(t):
    return (t - exp_beg_time.to_datetime64()) / np.timedelta64(1, 'h')


def horDecomp(da, name_m="mean", name_p="prime"):
    m = da.mean(dim="west_east").rename(name_m)
    p = (da - m).rename(name_p) 
    return m, p


# Loading data
print("Loading wrf dir: %s" % (args.input_dir,))






def loadData(input_dir):

    #print("Loading dir: ", input_dir)

    wsm = wrf_load_helper.WRFSimMetadata(
        start_datetime  = exp_beg_time,
        data_interval   = wrfout_data_interval,
        frames_per_file = args.frames_per_wrfout_file,
    )



    ds_nonavg = wrf_load_helper.loadWRFDataFromDir(
        wsm, 
        input_dir,
        beg_time = time_beg,
        end_time = time_end,
        prefix="wrfout_d01_",
        avg = None, #"ALL",
        verbose=False,
        inclusive="both",
    )


   
    ds_extra = wrf_preprocess.genAnalysis(ds_nonavg, wsm.data_interval)
    
    ds_extra = xr.merge( [ds_extra[varname] for varname in ["PRECIP", "TA", "QA", "TO", "QO", "VOR10", "DIV10", "DIV", "VOR", "U_T", "WND"]] ).mean(dim="time")
     
    #ACCRAIN = ds["RAINNC"] + ds["RAINC"]

    #PRECIP = ( ACCRAIN.isel(time=-1) - ACCRAIN.isel(time=0) ) / ( time_end - time_beg - wrfout_data_interval ).total_seconds()


    ds = wrf_load_helper.loadWRFDataFromDir(
        wsm, 
        input_dir,
        beg_time = time_beg,
        end_time = time_end,
        prefix="wrfout_d01_",
        avg = "ALL",
        verbose=False,
        inclusive="both",
    )

    ds = xr.merge([ds, ds_extra])


    WND10 = ((ds.U10**2 + ds.V10**2)**0.5).rename("WND10")
    ds = xr.merge([ds, WND10])

    has_TKE = "QKE" in ds


    # TKE variables
    def W2T(da_W):
        
        da_T = xr.zeros_like(ds["T"])
        
        da_W = da_W.to_numpy()
        da_T[:, :, :, :] = ( da_W[:, :-1, :, :] + da_W[:, 1:, :, :] ) / 2.0

        return da_T

    if args.tke_analysis == "TRUE" and has_TKE:
        DQKE_T = (2 * ds["DTKE"]).rename("DQKE_T")
        QSHEAR_T =   W2T(ds["QSHEAR"]).rename("QSHEAR_T")
        QBUOY_T  =   W2T(ds["QBUOY"]).rename("QBUOY_T")
        QWT_T    =   2*W2T(ds["QWT"]).rename("QWT_T")
        QDISS_T  = - W2T(ds["QDISS"]).rename("QDISS_T")

        QRES_T = (DQKE_T - (QSHEAR_T + QBUOY_T + QWT_T + QDISS_T)).rename("QRES_T")
        ds = xr.merge([ds, QSHEAR_T, QBUOY_T, QWT_T, QDISS_T, QRES_T, DQKE_T])


    # Virtual temperature
    THETA  = (300 + ds["T"]).rename("THETA")
    THETAV = THETA * (1 + 0.61*ds["QVAPOR"] - ds["QCLOUD"])
    THETAV = THETAV.rename("THETAV")

    ds = xr.merge([ds, THETAV, THETA])
    ds = ds.mean(dim=['time', 'south_north', 'south_north_stag'], keep_attrs=True)

    #ds_ref_stat.mean(dim=['time', 'south_north', 'south_north_stag', "west_east", "west_east_stag"], keep_attrs=True)
    ds_ref_stat = ds.mean(dim=["west_east", "west_east_stag"])


    ref_Z_W = (ds_ref_stat.PHB + ds_ref_stat.PH) / 9.81
    ref_Z_T = (ref_Z_W[1:] + ref_Z_W[:-1]) / 2

    Nx = ds.dims['west_east']
    Nz = ds.dims['bottom_top']

    X_sU = ds.DX * np.arange(Nx+1) / 1e3
    X_sT = (X_sU[1:] + X_sU[:-1]) / 2
    X_T = np.repeat(np.reshape(X_sT, (1, -1)), [Nz,], axis=0)
    X_W = np.repeat(np.reshape(X_sT, (1, -1)), [Nz+1,], axis=0)

    Z_W = (ds.PHB + ds.PH) / 9.81
    Z_T = (Z_W[1:, :] + Z_W[:-1, :]) / 2


    Z_W_idx_1km = np.argmin(np.abs(ref_Z_W.to_numpy() - 1e3)) 
    Z_W_idx_500m = np.argmin(np.abs(ref_Z_W.to_numpy() - 500.0)) 

    Z_T_idx_1km = np.argmin(np.abs(ref_Z_T.to_numpy() - 1e3)) 
    Z_T_idx_500m = np.argmin(np.abs(ref_Z_T.to_numpy() - 500.0)) 


    theta = ds.T + 300.0
    zeta = (ds.V[:, 1:] - ds.V[:, :-1]) / ds.DX


    theta_prime = ds["T"] - ds_ref_stat["T"]
    U_prime = ds["U"] - ds_ref_stat["U"]
    V_prime = ds["V"] - ds_ref_stat["V"]


    U_prime = U_prime.rename("U_prime")
    V_prime = V_prime.rename("V_prime")


    def ddz(da):
       
        da_np = da.to_numpy() 
        dfdz = xr.zeros_like(ds_ref_stat.PH)
       
        z_T = ref_Z_T.to_numpy() 
        dz = z_T[1:] - z_T[:-1]
        dfdz[1:-1] = (da_np[1:] - da_np[:-1]) / dz
        dfdz[0] = np.nan
        dfdz[-1] = np.nan

        return dfdz 
 

    NVfreq2 = (g0 / 300.0 * ddz(ds_ref_stat["THETAV"])).rename("NVfreq2")
    Nfreq2  = (g0 / 300.0 * ddz(ds_ref_stat["THETA"])).rename("Nfreq2")

       
    #NVfreq = ((g0 / 300.0 * ddz(ds_ref_stat["THETAV"]))**0.5).rename("NVfreq")
    #Nfreq = ((g0 / 300.0 * ddz(ds_ref_stat["THETA"]))**0.5).rename("Nfreq")

    #WND = (ds_ref_stat["U"]**2 +  ds_ref_stat["V"]**2)**0.5
    #WND = WND.rename("WND")

    ds_ref_stat = xr.merge([ds_ref_stat, NVfreq2, Nfreq2])

    print("Done")

    return(
        dict(
            ds=ds,
            ds_ref_stat=ds_ref_stat,
            ref_Z_W = ref_Z_W,
            ref_Z_T = ref_Z_T,
            X_sU = X_sU,
            X_sT = X_sT,
            X_T = X_T,
            X_W = X_W,
            Z_W = Z_W,
            Z_T = Z_T,
            Z_W_idx_1km = Z_W_idx_1km,
            Z_W_idx_500m = Z_W_idx_500m,
            Z_T_idx_1km = Z_T_idx_1km,
            Z_T_idx_500m = Z_T_idx_500m,

        )
    )

 

data = loadData(args.input_dir)

print("Done loading data.")

ds = data["ds"]
ds_ref_stat = data["ds_ref_stat"]

ref_Z_W = data["ref_Z_W"]
ref_Z_T = data["ref_Z_T"]
    
has_TKE = "QKE" in ds

print("Loading matplotlib...")
import matplotlib
if args.no_display:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
print("Done")

# =====================================================================

print("Generating vertical profile")

ncol = 5
nrow = 1

w = [1.5, 1.5, 1.5, 1.5, 1.5]

if args.tke_analysis == "TRUE":

    ncol += 2 
    w = w + [1, 2,]

figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = w,
    h = [4, ],
    wspace = 1.0,
    hspace = 1.0,
    w_left = 1.0,
    w_right = 1.0,
    h_bottom = 1.0,
    h_top = 1.0,
    ncol = ncol,
    nrow = nrow,
)

fig, ax = plt.subplots(
    nrow, ncol,
    figsize=figsize,
    subplot_kw=dict(aspect="auto"),
    gridspec_kw=gridspec_kw,
    constrained_layout=False,
    squeeze=False,
)


if args.overwrite_title == "":
    fig.suptitle("%sTime: %.2f ~ %.2f hr" % (args.extra_title, relTimeInHrs(time_beg), relTimeInHrs(time_end),))
    
else:
    fig.suptitle(args.overwrite_title)


iii = 0
# Temperature
_ax = ax[0, iii]
_ax.plot(ds_ref_stat["THETA"], ref_Z_T, 'k-', label="$\\overline{\\theta}$")
_ax.plot(ds_ref_stat["THETAV"], ref_Z_T, 'r--', label="$\\overline{\\theta_v}$")
_ax.set_title("(%s) $  \\overline{\\theta}$ (-), $ \\overline{\\theta}_v$ (--)" % (args.thumbnail_numbering[args.thumbnail_skip + iii],))
_ax.set_xlabel("[ $\\mathrm{K}$ ]")
_ax.set_xlim(args.THETA_rng)
#_ax.legend(loc="upper right")
iii += 1

# Stability
_ax = ax[0, iii]
_ax.plot(ds_ref_stat["Nfreq2"] * 1e4, ref_Z_W, 'k-', label="$N^2$")
_ax.plot(ds_ref_stat["NVfreq2"] * 1e4, ref_Z_W, 'r--', label="$N^2_v$")
_ax.set_title("(%s) $ N^2$ (-), $ N^2_v$ (--)" % (args.thumbnail_numbering[args.thumbnail_skip + iii],))
_ax.set_xlabel("[ $\\times 10^{-4} \\mathrm{s}^{-2}$ ]")
_ax.set_xlim(args.Nfreq2_rng)
#_ax.legend(loc="upper right")
iii += 1

# U, V
_ax = ax[0, iii]
_ax.plot(ds_ref_stat["WND"], ref_Z_T, linestyle="-", color="black")
_ax.plot(ds_ref_stat["U"], ref_Z_T, linestyle="--", color="blue")#, ref_Z_T)
_ax.plot(ds_ref_stat["V"], ref_Z_T, linestyle=":", color="red")#, ref_Z_T)

_ax.set_title("(%s) $ U $ (-), $ \\overline{u}$ (--), $ \\overline{v}$ (..)" % (args.thumbnail_numbering[args.thumbnail_skip + iii],))
_ax.set_xlabel("[ $\\mathrm{m} \\, / \\, \\mathrm{s}$ ]")
_ax.set_xlim(args.U_rng)
iii += 1


# WND
"""
_ax = ax[0, iii]
_ax.plot(ds_ref_stat["WND"], ref_Z_T, color="black")

_ax.set_title("(%s) $ \\left| \\overline{\\vec{U}} \\right|$" % (args.thumbnail_numbering[args.thumbnail_skip + iii],))
_ax.set_xlabel("[ $\\mathrm{m} \\, / \\, \\mathrm{s}$ ]")
_ax.set_xlim(args.U_rng)
iii += 1
"""

# TKE
_ax = ax[0, iii]
_ax.plot(ds_ref_stat["QKE"]/2, ref_Z_T, color="black")
_ax.set_title("(%s) $ \\overline{\\mathrm{TKE}}$" % (args.thumbnail_numbering[args.thumbnail_skip + iii],))
_ax.set_xlabel("[ $\\mathrm{m}^2 \\, / \\, \\mathrm{s}^2$ ]")
_ax.set_xlim(args.TKE_rng)
iii+=1



# Q
_ax = ax[0, iii]
_ax.plot(ds_ref_stat["QVAPOR"] * 1e3, ref_Z_T, color="black")

_ax.set_title("(%s) $ \\overline{Q}_\\mathrm{vapor}$" % (args.thumbnail_numbering[args.thumbnail_skip + iii],))
_ax.set_xlabel("[ $\\mathrm{g} \\, / \\, \\mathrm{kg}$ ]")
_ax.set_xlim(args.Q_rng)
iii += 1



if args.tke_analysis == "TRUE":
    # TKE
    _ax = ax[0, iii]
    _ax.plot(ds_ref_stat["QKE"]/2, ref_Z_T, 'k-')
    _ax.set_title("(%s) $ \\overline{\\mathrm{TKE}}$" % (args.thumbnail_numbering[args.thumbnail_skip + iii],))
    _ax.set_xlabel("[ $\\mathrm{m}^2 \\, / \\, \\mathrm{s}^2$ ]")
    _ax.set_xlim(args.TKE_rng)
    iii+=1

    # TKE budget
    _ax = ax[0, iii]

    _ax.set_title("(%s) TKE budget" % (args.thumbnail_numbering[args.thumbnail_skip + iii],))
    _ax.plot(ds_ref_stat["DQKE_T"]  / 2, ref_Z_T,   color="black", linestyle='-', label="$ \\frac{\\partial q}{\\partial t}$")
    _ax.plot(ds_ref_stat["QSHEAR_T"]/2, ref_Z_T, color="red", linestyle='-',   label="$q_{sh}$")
    _ax.plot(ds_ref_stat["QBUOY_T"]/2, ref_Z_T,  color="blue", linestyle='-',  label="$q_{bu}$")
    _ax.plot(ds_ref_stat["QWT_T"]/2 , ref_Z_T,    color="red", linestyle='--',  label="$q_{vt}$")
    _ax.plot(ds_ref_stat["QDISS_T"]/2, ref_Z_T,  color="blue", linestyle='--', label="$q_{ds}$")
    _ax.plot(ds_ref_stat["QRES_T"]/2, ref_Z_T,   color="#aaaaaa", linestyle='--',label="$res$")

    #_ax.plot(ds_ref_stat["QWT_T"]/2 + ds_ref_stat["QADV_T"]/2, ref_Z_T,    color="red", linestyle='--',  label="$q_{vt} + q_{adv}$")

    _ax.set_xlabel("[ $\\mathrm{m}^2 \\, / \\, \\mathrm{s}^2$ ]")
    _ax.set_xlim(args.DTKE_rng)

    _ax.legend(loc="center right")
    iii+=1

for _ax in ax[0, :]:
    trans = transforms.blended_transform_factory(_ax.transAxes, _ax.transData)
    #_ax.plot([0, 1], [ds_base_ref_stat["PBLH"].to_numpy()]*2, color="magenta", linestyle=":", transform=trans)
    _ax.plot([0, 1], [ds_ref_stat["PBLH"].to_numpy()]*2, color="magenta", linestyle="--", transform=trans)



for i, _ax in enumerate(ax[0, :]):

    _ax.set_ylim(args.z_rng)
     
    _ax.set_ylabel("$z$ [ km ]")
    yticks = np.array(_ax.get_yticks())
    _ax.set_yticks(yticks, ["%.1f" % _y for _y in yticks/1e3])

    _ax.grid(visible=True, which='major', axis='both')


if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()


