import xarray as xr
import pandas as pd
import numpy as np
import argparse
import load_helper
import tool_fig_config
import diagnostics
import datetime
import wrf_load_helper 

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-dir', type=str, help='Input directory.', required=True)
parser.add_argument('--output', type=str, help='Output filename in png.', default="")
parser.add_argument('--extra-title', type=str, help='Title', default="")
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
parser.add_argument('--x-rng', type=float, nargs=2, help='The plotted height rng in kilometers', default=[None, None])
parser.add_argument('--U10-rng', type=float, nargs=2, help='The plotted surface wind in m/s', default=[None, None])

args = parser.parse_args()

print(args)

exp_beg_time = pd.Timestamp(args.exp_beg_time)
wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)

time_beg = exp_beg_time + pd.Timedelta(minutes=args.time_rng[0])
time_end = exp_beg_time + pd.Timedelta(minutes=args.time_rng[1])


# Loading data
print("Loading wrf dir: %s" % (args.input_dir,))

wsm = wrf_load_helper.WRFSimMetadata(
    start_datetime  = exp_beg_time,
    data_interval   = wrfout_data_interval,
    frames_per_file = args.frames_per_wrfout_file,
)


ds = wrf_load_helper.loadWRFDataFromDir(
    wsm, 
    args.input_dir,
    beg_time = time_beg,
    end_time = time_end,
    prefix="wrfout_d01_",
    avg=False,
    verbose=False,
    inclusive="both",
)

ds = ds.mean(dim=['south_north', 'south_north_stag'], keep_attrs=True)
print("Done")

Nx = ds.dims['west_east']
Nz = ds.dims['bottom_top']

X_sU = ds.DX * np.arange(Nx+1) / 1e3
X_sT = (X_sU[1:] + X_sU[:-1]) / 2
X_T = np.repeat(np.reshape(X_sT, (1, -1)), [Nz,], axis=0)

t = ds.coords["time"]



print("Loading matplotlib...")
import matplotlib
if args.no_display:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
print("Done")

fig, ax = plt.subplots(
    1, 1,
    figsize=(10, 6),
    subplot_kw=dict(aspect="auto"),
    constrained_layout=False,
)

time_fmt="%y/%m/%d %Hh"
fig.suptitle("%sTime: %s ~ %s" % (args.extra_title, time_beg.strftime(time_fmt), time_end.strftime(time_fmt)))

U10_levs = np.linspace(4, 8, 9)

mappable1 = ax.contourf(X_sT, t, ds.U10, levels=U10_levs, cmap="gnuplot", extend="both")
cax = tool_fig_config.addAxesNextToAxes(fig, ax, "right", thickness=0.03, spacing=0.05)
cbar1 = plt.colorbar(mappable1, cax=cax, orientation="vertical")

ax.grid()
ax.set_xlabel("[km]")
ax.set_xlim(np.array(args.x_rng))

if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()

