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
parser.add_argument('--thumbnail-numbering', type=str, help='Thumbnail numbering', default="abcdefghijklmn")
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
parser.add_argument('--diag-press-lev', type=float, help='A selected pressure level to output diagnostics in hPa.', default=850.0)
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

time_beg = exp_beg_time + pd.Timedelta(hours=args.time_rng[0])
time_end = exp_beg_time + pd.Timedelta(hours=args.time_rng[1])

# Enclosed means the shaded time interval. It is used to estimate
# the pressure. We simply want to remove the spin-up time (probably the first two hours)
enclosed_time_beg = exp_beg_time + pd.Timedelta(hours=args.enclosed_time_rng[0])
enclosed_time_end = exp_beg_time + pd.Timedelta(hours=args.enclosed_time_rng[1])


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



PH_W = ds.PHB + ds.PH
Z_W = PH_W / 9.81
Z_T = (Z_W[1:, :] + Z_W[:-1, :]) / 2

P_T = ds.PB + ds.P

# Estimate selected eta

print("## Estimate the desired eta level to match the wanted pressure level...")

# Need to subset the desired time
print("Enclosed time: [%s, %s]" % (enclosed_time_beg, enclosed_time_end,))
ds_subset = ds.where(
    (ds.time >= enclosed_time_beg) & (ds.time < enclosed_time_end)
)

P_T_subset = ds_subset.PB + ds_subset.P
eta_mean_W = ds_subset["ZNW"].mean(dim=["time",])
P_hmean = P_T_subset.mean(dim=["west_east", "time"])

selected_z_idx = np.argmin(np.abs(P_hmean.to_numpy() - args.diag_press_lev*1e2))
selected_eta = eta_mean_W.isel(bottom_top_stag=selected_z_idx)

print("The `--diag-press-lev` = %.2fhPa. The estimate z_idx=%d and the found eta = %f" % (args.diag_press_lev, selected_z_idx, selected_eta))


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


dWdt = ( ds.W.isel(time=slice(1, None, None), bottom_top_stag=selected_z_idx).to_numpy() - ds.W.isel(time=slice(0, -1, None), bottom_top_stag=selected_z_idx).to_numpy() ) / delta_t[:, None]

dU10dt = ( ds.U10.isel(time=slice(1, None, None)).to_numpy() - ds.U10.isel(time=slice(0, -1, None)).to_numpy() ) / delta_t[:, None]


max_absdU10dt = np.amax(np.abs(dU10dt), axis=1)
rms_dU10dt = np.mean(dU10dt**2, axis=1)**0.5

max_absdWdt = np.amax(np.abs(dWdt), axis=1)
rms_dWdt    = np.mean(dWdt**2, axis=1)**0.5


ratio_rms_dU10dt = np.zeros_like(rms_dU10dt)
ratio_rms_dU10dt[1:] = (rms_dU10dt[1:] - rms_dU10dt[:-1]) / rms_dU10dt[1:]
ratio_rms_dU10dt[0] = np.nan


data = dict(
    dU10dt = dict(max_abs = max_absdU10dt, rms = rms_dU10dt,),
    dWdt   = dict(max_abs = max_absdWdt, rms = rms_dWdt,),
)

plot_infos = dict(

    dU10dt = dict(
        factor = 1e4,
        label = "$\\partial U_\\mathrm{10m} / \\partial t$",
        unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
        ylim = [0, 8],
    ),

    dWdt = dict(
        factor = 1e7,
        label = "$\\partial w_{%d \\mathrm{hPa}} / \\partial t$" % (args.diag_press_lev,),
        unit = "$ \\times 10^{-5} \\, \\mathrm{cm} / \\mathrm{s}^2 $",
        ylim = [0, 10],
    ),

)


#plot_varnames = ["dU10dt", "dWdt"]
plot_varnames = ["dWdt"]

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

ncol = 1
nrow = len(plot_varnames)


figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = 4,
    h = 3,
    wspace = 1.0,
    hspace = 0.7,
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
    sharex=True,
)

#time_fmt="%y/%m/%d %Hh"
#fig.suptitle("%sTime: %s ~ %s" % (args.extra_title, time_beg.strftime(time_fmt), time_end.strftime(time_fmt)))

fig.suptitle("%sTime: %d ~ %d hr" % (args.extra_title, relTimeInHrs(time_beg), relTimeInHrs(time_end),))


for i, varname in enumerate(plot_varnames):
    _ax = ax.flatten()[i]
    vardata = data[varname]
    plot_info = plot_infos[varname]

    factor = plot_info["factor"] if "factor" in plot_info else 1.0
    ylim   = plot_info["ylim"] if "ylim" in plot_info else [None, None]

    if args.enclosed_time_rng is not None:
        blended_trans = matplotlib.transforms.blended_transform_factory(_ax.transData, _ax.transAxes)
        rect = matplotlib.patches.Rectangle([args.enclosed_time_rng[0], 0], args.enclosed_time_rng[1] - args.enclosed_time_rng[0], 1, transform=blended_trans, edgecolor=None, facecolor=(0.9, 0.9, 0.9))

        _ax.add_patch(rect)

    _ax.plot(mid_t_rel, vardata["max_abs"] * factor, "b-", label="max_abs")
    _ax.plot(mid_t_rel, vardata["rms"] * factor, "r-", label="rms")

    _ax.set_ylim(ylim)

    _ax.set_title("({thumbnail_numbering:s}) {label:s} [{unit:s}]".format(
        thumbnail_numbering = args.thumbnail_numbering[i],
        label = plot_info["label"],
        unit  = plot_info["unit"]),
    )

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

