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
from interpolate_Z import interpolateZ

def mavg(a, axis, half_window_size=0):

    #print("Smooth over W with rolling = ", args.part1_x_rolling)
    window = 2*half_window_size+1

    _a = np.zeros_like(a)
    for shift in range(- half_window_size , half_window_size+1):
        _a += np.roll(a, shift, axis=axis)

    _a /= window

    return _a




parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-dir', type=str, help='Input directory.', required=True)
parser.add_argument('--input-dir-base', type=str, help='Input directory for base.', default=None)
parser.add_argument('--output', type=str, help='Output filename in png.', default="")
parser.add_argument('--extra-title', type=str, help='Title', default="")
parser.add_argument('--overwrite-title', type=str, help='If set then title will be set to this.', default="")

parser.add_argument('--no-display', action="store_true")
parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
parser.add_argument('--height-mode', type=str, help="If height-mode = `grid`, then heights mean grid index. If `Z`, then heights mean the phyiscal height above surface.", required=True)
parser.add_argument('--heights', type=float, nargs="+", help="Height to be plotted in meter.", required=True)
parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)

parser.add_argument('--thumbnail-skip', type=int, help='Skip of thumbnail numbering.', default=0)
parser.add_argument('--thumbnail-numbering', type=str, help='Skip of thumbnail numbering.', default="abcdefghijklmn")
parser.add_argument('--x-rng', type=float, nargs=2, help='The plotted height rng in kilometers', default=[None, None])
parser.add_argument('--xmavg-half-window-size', type=int, help='The plotted height rng in kilometers', default=0)
parser.add_argument('--f0', type=float, help='The plotted height rng in kilometers', required=True)



args = parser.parse_args()

print(args)

base_exists = args.input_dir_base is not None    

exp_beg_time = pd.Timestamp(args.exp_beg_time)
wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)

time_beg = exp_beg_time + pd.Timedelta(minutes=args.time_rng[0])
time_end = exp_beg_time + pd.Timedelta(minutes=args.time_rng[1])

def relTimeInHrs(t):
    return (t - exp_beg_time.to_datetime64()) / np.timedelta64(1, 'h')


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
    ).mean(dim=["south_north", 'south_north_stag'], keep_attrs=True)

    ds = wrf_preprocess.genDivAnalysis(ds_nonavg, wsm.data_interval, f0=args.f0)
    
    needed_vars = list(ds.keys())
    ds = xr.merge([
        ds,
        ds_nonavg["PH"],
        ds_nonavg["PHB"],
        ds_nonavg["T"],
    ])

    print(ds)

    if args.height_mode == "Z":
        ds = interpolateZ(ds, needed_vars, z=args.heights)
    elif args.height_mode == "grid":
        grid_array = np.array(args.heights).astype(int)
        ds = ds.isel(bottom_top=grid_array)
        ds = ds.assign_coords({"bottom_top": grid_array})
    
    ds = ds.mean(dim="time")

    return ds

 

ds = loadData(args.input_dir)

if base_exists:
    ds_base = loadData(args.input_dir_base)
    diff_ds = ds - ds_base
else:
    diff_ds = ds

if args.height_mode == "Z":
    vertical_coord = "Z"
elif args.height_mode == "grid":
    vertical_coord = "bottom_top"



print("Done loading data.")


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
nrow = 1 + len(args.heights)

w = [6,]

figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = w,
    h = [4] + [4,] * len(args.heights),
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


thumbnail_numberings = args.thumbnail_numbering[args.thumbnail_skip:]
ax_cnt = 0

def nextAxes():

    global ax_cnt 

    _ax = ax[ax_cnt, 0]
    thumbnail_numbering = thumbnail_numberings[ax_cnt]
    
    ax_cnt += 1

    return _ax, thumbnail_numbering


_ax, _thumbnail_numbering = nextAxes()

def vertical_expression(mode, z):
    if mode == "Z":
        return "%d m" % (z,)
    elif mode == "grid":
        return "z-index=%d" % (z,)


for i, z in enumerate(diff_ds.coords[vertical_coord]):
  
    _ds = diff_ds.sel(**{vertical_coord: z})

    print(_ds)
    _ax.plot(_ds.coords["west_east"], _ds["DIV"], label=vertical_expression(args.height_mode, z))
    #"%d m" % (z,))#color="magenta", linestyle=":")
    



_ax.set_title("(%s) Divergence" % (_thumbnail_numbering,))
_ax.set_ylabel("[ $ \\mathrm{s}^{-1}$ ]", color="black")
    
_ax.legend()


for i, z in enumerate(diff_ds.coords[vertical_coord]):
  
    _ds = diff_ds.sel(**{vertical_coord: z})

    _ax, _thumbnail_numbering = nextAxes()
     
    for varname, label in [
        ["dDIVdt", "$ \\mathrm{d}\\delta/\\mathrm{d}t$"],
#        ["dDIVdt_est", "$ \\mathrm{d}\\delta/\\mathrm{d}t$ est"],
        ["VM_term", "VM"],
#        ["VM_term_indirect", "VM ind"],
        ["BPG_term", "BPG"],
        ["DIV_term", "$-\\delta^2$"],
        ["VOR_term", "$f \\cdot \\zeta$"],
        ["DEFO_term", "$- w_x u_z$"],
    ]:

        d = _ds[varname].to_numpy()
        plot_data = mavg(d, axis=0, half_window_size=args.xmavg_half_window_size)

        _ax.plot(_ds.coords["west_east"], plot_data, label=label)
    

    _ax.set_title("(%s) %s" % (_thumbnail_numbering, vertical_expression(args.height_mode, z)))
    _ax.set_ylabel("[ $ \\mathrm{s}^{-2}$ ]", color="black")
    

    _ax.legend()


for i, _ax in enumerate(ax[:, 0]):
    
    _ax.set_xlabel("$x$ [km]")
    _ax.set_xlim(np.array(args.x_rng))
    _ax.grid(visible=True, which='major', axis='both')

if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()

