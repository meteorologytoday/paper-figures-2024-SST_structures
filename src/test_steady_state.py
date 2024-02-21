import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
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
parser.add_argument('--enclosed-time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time. This interval will be shaded by a gray background", default=None)
parser.add_argument('--coarse-grained-time', type=int, help='The number of seconds to average the wrfout data. It has to be a multiple of `--wrfout-data-interval`.', default=None)

args = parser.parse_args()

print(args)

exp_beg_time = pd.Timestamp(args.exp_beg_time)
wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)

# Determine if `--coarse-grained-time` is valid
if args.coarse_grained_time is None: 
    args.coarse_grained_time = args.wrfout_data_interval
    
coarse_grained_time = pd.Timedelta(seconds=args.coarse_grained_time) 

if (coarse_grained_time / wrfout_data_interval) % 1 != 0:
    raise Exception("`--coarse-grained-time` is not a multiple of `--wrfout-data-interval`") 

time_beg = exp_beg_time + pd.Timedelta(minutes=args.time_rng[0])
time_end = exp_beg_time + pd.Timedelta(minutes=args.time_rng[1])

def relTimeInHrs(t):
    return (t - exp_beg_time.to_datetime64()) / np.timedelta64(1, 'h')

wsm = wrf_load_helper.WRFSimMetadata(
    start_datetime  = exp_beg_time,
    data_interval   = wrfout_data_interval,
    frames_per_file = args.frames_per_wrfout_file,
)


# Loading data
print("Loading wrf dir: %s" % (args.input_dir,))
ds = wrf_load_helper.loadWRFDataFromDir(
    wsm, 
    args.input_dir,
    beg_time = time_beg,
    end_time = time_end,
    prefix="wrfout_d01_",
    avg=coarse_grained_time,
    verbose=False,
    inclusive="left",
)

ds = ds.mean(dim=['south_north', 'south_north_stag'], keep_attrs=True)
print(ds)
print("Done loading data.")

Nx = ds.dims['west_east']
Nz = ds.dims['bottom_top']

X_sU = ds.DX * np.arange(Nx+1) / 1e3
X_sT = (X_sU[1:] + X_sU[:-1]) / 2
X_T = np.repeat(np.reshape(X_sT, (1, -1)), [Nz,], axis=0)

t = ds.coords["time"].to_numpy()

t_rel = relTimeInHrs(t)


delta_t = t[1:] - t[:-1]
mid_t = t[:-1] + 0.5 * delta_t
mid_t_rel = relTimeInHrs(mid_t)

if delta_t.dtype != "timedelta64[ns]":
    raise Exception("Type of time is not timedelta64[ns]")

delta_t = delta_t.astype(np.float64) / 1e9
#print(delta_t)

print("delta_t shape = ", delta_t.shape, " ; of type: ", delta_t.dtype)
print(ds.U10.to_numpy().shape)
print(ds.U10.isel(time=slice(1, None, None)).to_numpy().shape)
print(ds.U10.isel(time=slice(None, -1, None)).to_numpy().shape)

dWdt = ( ds.W.isel(time=slice(1, None, None), bottom_top_stag=20).to_numpy() - ds.W.isel(time=slice(0, -1, None), bottom_top_stag=20).to_numpy() ) / delta_t[:, None]

dU10dt = ( ds.U10.isel(time=slice(1, None, None)).to_numpy() - ds.U10.isel(time=slice(0, -1, None)).to_numpy() ) / delta_t[:, None]


max_absdU10dt = np.amax(np.abs(dU10dt), axis=1)
rms_dU10dt = np.mean(dU10dt**2, axis=1)**0.5

max_absdWdt = np.amax(np.abs(dWdt), axis=1)
rms_dWdt    = np.mean(dWdt**2, axis=1)**0.5


ratio_rms_dU10dt = np.zeros_like(rms_dU10dt)
ratio_rms_dU10dt[1:] = (rms_dU10dt[1:] - rms_dU10dt[:-1]) / rms_dU10dt[1:]
ratio_rms_dU10dt[0] = np.nan


print("Shape of max_absdU10dt: ", max_absdU10dt.shape)
print("Shape of rms_dU10dt: ", rms_dU10dt.shape)

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
    figsize=(6, 4),
    subplot_kw=dict(aspect="auto"),
    constrained_layout=False,
    sharex=True,
    squeeze=False,
)

#time_fmt="%y/%m/%d %Hh"
#fig.suptitle("%sTime: %s ~ %s" % (args.extra_title, time_beg.strftime(time_fmt), time_end.strftime(time_fmt)))

fig.suptitle("%sTime: %d ~ %d hr" % (args.extra_title, relTimeInHrs(time_beg), relTimeInHrs(time_end),))


_ax = ax.flatten()[0]

if args.enclosed_time_rng is not None:
    blended_trans = matplotlib.transforms.blended_transform_factory(_ax.transData, _ax.transAxes)
    rect = matplotlib.patches.Rectangle([args.enclosed_time_rng[0], 0], args.enclosed_time_rng[1] - args.enclosed_time_rng[0], 1, transform=blended_trans, edgecolor=None, facecolor=(0.9, 0.9, 0.9))

    _ax.add_patch(rect)

_ax.plot(mid_t_rel, max_absdU10dt * 1e4, "b-", label="max_abs")
_ax.plot(mid_t_rel, rms_dU10dt * 1e4, "r-", label="rms")




#_twinx = _ax.twinx()
#_twinx.plot(mid_t_rel, ratio_rms_dU10dt, "r--")

_ax.set_ylabel("$\\frac{\\partial U_\\mathrm{10m}}{\\partial t}$ [$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $]")

#_ax = ax.flatten()[1]
#_ax.plot(mid_t, max_absdWdt, "b-", label="max")
#_ax.plot(mid_t, rms_dWdt, "r-", label="rms")

total_time = relTimeInHrs(time_end)

for _ax in ax.flatten():
    _ax.legend()
    _ax.grid()
    _ax.set_xlabel("[ hr ]")
    _ax.set_xticks(12 * np.arange(np.ceil(total_time / 12)+1))

if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()

