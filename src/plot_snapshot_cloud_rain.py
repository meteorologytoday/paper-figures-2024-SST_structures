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

parser.add_argument('--z-rng', type=float, nargs=2, help='The plotted height rng in meters.', default=[0, 5000.0])
parser.add_argument('--x-rng', type=float, nargs=2, help='The plotted height rng in kilometers', default=[None, None])

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
    ds_extra = xr.merge( [ds_extra[varname] for varname in ["PRECIP",]] ).mean(dim="time")
     
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

    ds = xr.merge([ds, ds_extra]).mean(dim=['time', 'south_north', 'south_north_stag'], keep_attrs=True)

    Nx = ds.dims['west_east']
    Nz = ds.dims['bottom_top']

    X_sU = ds.DX * np.arange(Nx+1) / 1e3
    X_sT = (X_sU[1:] + X_sU[:-1]) / 2
    X_T = np.repeat(np.reshape(X_sT, (1, -1)), [Nz,], axis=0)
    X_W = np.repeat(np.reshape(X_sT, (1, -1)), [Nz+1,], axis=0)
    Z_W = (ds.PHB + ds.PH) / 9.81
    Z_T = (Z_W[1:, :] + Z_W[:-1, :]) / 2

    ds_ref_stat = ds.mean(dim=["west_east", "west_east_stag",])
    ref_Z_W = ( (ds_ref_stat.PHB + ds_ref_stat.PH) / 9.81 ).to_numpy()
    ref_Z_T = (ref_Z_W[1:] + ref_Z_W[:-1]) / 2


    Z_W_idx_1km = np.argmin(np.abs(ref_Z_W - 1e3)) 
    Z_W_idx_500m = np.argmin(np.abs(ref_Z_W - 500.0)) 
    Z_T_idx_1km = np.argmin(np.abs(ref_Z_T - 1e3)) 
    Z_T_idx_500m = np.argmin(np.abs(ref_Z_T - 500.0)) 

    return dict(
        ds=ds,
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

 

data = loadData(args.input_dir)

print("Done loading data.")


ds = data["ds"]
X_sU = data["X_sU"]
X_sT = data["X_sT"]
X_T = data["X_T"]
X_W = data["X_W"]
Z_W = data["Z_W"]
Z_T = data["Z_T"]

qr = ds["QRAIN"].to_numpy()
print("QRAIN mean : ", np.mean(qr))
print("QRAIN max  : ", np.amax(qr))
print("QRAIN std  : ", np.std(qr))

    
print("Loading matplotlib...")
import matplotlib
if args.no_display:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
print("Done")

# ======= FIRST PART ========

ncol = 1
nrow = 4

w = [6,]

figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = w,
    h = [4,] + [4/3,] * (nrow-1),
    wspace = 1.0,
    hspace = 1.0,
    w_left = 1.0,
    w_right = 1.5,
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

#time_fmt="%y/%m/%d %Hh"

if args.overwrite_title == "":
    fig.suptitle("%sTime: %d ~ %d hr" % (args.extra_title, relTimeInHrs(time_beg), relTimeInHrs(time_end),))
    
else:
    fig.suptitle(args.overwrite_title)


thumbnail_numberings = "abcdefghijklmn"
ax_cnt = 0

def nextAxes():

    global ax_cnt 

    _ax = ax[ax_cnt, 0]
    thumbnail_numbering = thumbnail_numberings[ax_cnt]
    
    ax_cnt += 1

    return _ax, thumbnail_numbering


cmap_diverge = cmocean.cm.balance 
cmap_linear = cmocean.cm.matter

# Version 1: shading = TKE, contour = W

_ax, _thumbnail_numbering = nextAxes()

# Version 2: shading = W, contour = TKE
mappable1 = _ax.contourf(X_T, Z_T, ds["QRAIN"] * 1e6, levels = np.arange(0.5, 10, 1.0), cmap=cmap_linear, extend="max")
cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
cbar0 = plt.colorbar(mappable1, cax=cax, orientation="vertical")
cbar0.set_label("$Q_{\\mathrm{rain}}$ [$ \\times 10^{3} \\mathrm{g} / \\mathrm{kg}$]")
_ax.plot(X_sT, ds["PBLH"], color="magenta", linestyle="--")

_ax.set_title("(%s) $Q_{\\mathrm{rain}}$" % (_thumbnail_numbering,))

# Begin the line plots
_ax, _thumbnail_numbering = nextAxes()
_ax.plot(X_sT, ds["QRAIN"].isel(bottom_top=data["Z_T_idx_500m"]) * 1e6, color='black', linestyle="-", label="Precip")
_ax.set_title("(%s) $Q_{\\mathrm{rain}}$ at 500m [$ \\times 10^{3} \\mathrm{g} / \\mathrm{kg} $] " % (_thumbnail_numbering,))

_ax, _thumbnail_numbering = nextAxes()
_ax.plot(X_sT, ds["W"].isel(bottom_top_stag=data["Z_W_idx_500m"]) * 1e2, color='black', linestyle="-", label="Precip")
_ax.set_title("(%s) W at 500m [$ \\mathrm{cm} / \\mathrm{s} $] " % (_thumbnail_numbering,))


_ax, _thumbnail_numbering = nextAxes()
_ax.plot(X_sT, ds["PRECIP"] * 3600.0, color='black', linestyle="-", label="Precip")
_ax.set_title("(%s) Precip [$ \\mathrm{mm} / \\mathrm{hr} $] " % (_thumbnail_numbering,))

for i, _ax in enumerate(ax[:, 0]):
    
    _ax.set_xlabel("$x$ [km]")
    _ax.set_xlim(np.array(args.x_rng))

    if i == 0:
        
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

