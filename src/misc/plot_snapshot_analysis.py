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
parser.add_argument('--SST-rng', type=float, nargs=2, help='Title', default=[14.5, 16.5])
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
parser.add_argument('--z-rng', type=float, nargs=2, help='The plotted height rng in meters.', default=[0, 1200.0])
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

ds = ds.mean(dim=['time', 'south_north', 'south_north_stag'], keep_attrs=True)
print("Done")

Nx = ds.dims['west_east']
Nz = ds.dims['bottom_top']

X_sU = ds.DX * np.arange(Nx+1) / 1e3
X_sT = (X_sU[1:] + X_sU[:-1]) / 2
X_T = np.repeat(np.reshape(X_sT, (1, -1)), [Nz,], axis=0)
X_W = np.repeat(np.reshape(X_sT, (1, -1)), [Nz+1,], axis=0)

Z_W = (ds.PHB + ds.PH) / 9.81
Z_T = (Z_W[1:, :] + Z_W[:-1, :]) / 2

zerodegC = 273.15
theta = ds.T + 300.0
zeta = (ds.V[:, 1:] - ds.V[:, :-1]) / ds.DX
SST = ds.TSK - zerodegC

delta = ds.TH2 - ds.TSK



print("Loading matplotlib...")
import matplotlib
if args.no_display:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
print("Done")


fig, ax = plt.subplots(
    3, 1,
    figsize=(8, 9),
    subplot_kw=dict(aspect="auto"),
    gridspec_kw=dict(height_ratios=[3, 1, 1], right=0.8),
    constrained_layout=False,
    sharex=True,
)

time_fmt="%y/%m/%d %Hh"
fig.suptitle("%sTime: %s ~ %s" % (args.extra_title, time_beg.strftime(time_fmt), time_end.strftime(time_fmt)))


u_levs = np.linspace(4, 18, 15)
v_levs = np.linspace(0, 5, 21)
w_levs = np.linspace(-5, 5, 21) / 10
theta_levs = np.arange(273, 500, 2)

mappable1 = ax[0].contourf(X_W, Z_W, ds.W * 1e2, levels=w_levs, cmap="bwr", extend="both")

cs = ax[0].contour(X_T, Z_T, theta, levels=theta_levs, colors='k')
plt.clabel(cs)
cax = tool_fig_config.addAxesNextToAxes(fig, ax[0], "right", thickness=0.03, spacing=0.05)
cbar1 = plt.colorbar(mappable1, cax=cax, orientation="vertical")



U10_mean = np.mean(ds.U10)
V10_mean = np.mean(ds.V10)
ax[1].plot(X_sT, ds.U10 - U10_mean, color="black", label="$U_{\\mathrm{10m}} - \\overline{U}_{\\mathrm{10m}}$")
ax[1].plot(X_sT, ds.V10 - V10_mean, color="red",   label="$V_{\\mathrm{10m}} - \\overline{V}_{\\mathrm{10m}}$")



for _ax in ax[0:1].flatten():
    _ax.plot(X_sT, ds.PBLH, color="pink", linestyle="--")


# SST
ax[2].plot(X_sT, SST, color='blue', label="SST")
ax[2].plot(X_sT, ds.T2 - zerodegC, color='red', label="$T_{\\mathrm{2m}}$")


ax[0].set_title("W [$\\mathrm{cm} / \\mathrm{s}$]")
ax[1].set_title("$\\left( \\overline{U}_{\\mathrm{10m}}, \\overline{V}_{\\mathrm{10m}}\\right) = \\left( %.2f, %.2f \\right)$" % (U10_mean, V10_mean,))
ax[1].legend()
ax[2].legend()
#ax[1].set_title("U [$\\mathrm{m} / \\mathrm{s}$]")
#ax[2].set_title("V [$\\mathrm{m} / \\mathrm{s}$]")

for _ax in ax[0:1].flatten():
    _ax.set_ylim(args.z_rng)
    _ax.set_ylabel("z [ m ]")

ax[1].set_ylim(args.U10_rng)
ax[1].set_ylabel("[ $ \\mathrm{m} / \\mathrm{s}$ ]", color="black")
ax[2].set_ylabel("[ $ \\mathrm{K}$ ]", color="black")

for _ax in ax.flatten():
    _ax.grid()
    _ax.set_xlabel("[km]")
    _ax.set_xlim(np.array(args.x_rng))


if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()

