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

# Compute TOA and QOA
TO     = ds["TSK"].rename("TO")
TA     = (300.0 + ds["T"].isel(bottom_top=0)).rename("TA")
TOA    = (TO-TA).rename("TOA")

# Bolton (1980)
E1 = 0.6112e3 * np.exp(17.67 * (ds["TSK"] - 273.15) / (ds["TSK"] - 29.65) )
QSFCMR = 0.622 * E1 / (ds["PSFC"] - E1)
QO = QSFCMR.rename("QO")
QA = ds["QVAPOR"].isel(bottom_top=0).rename("QA")

QOA = QO - QA
QOA = QOA.rename("QOA")

W_b = ds["W"].isel(bottom_top_stag=slice(0, -1))
W_t = ds["W"].isel(bottom_top_stag=slice(1, None))

W = ds["QVAPOR"].copy()

W[:, :, :] = (W_b.to_numpy() + W_t.to_numpy()) / 2

z_idx = 40

WRHO = W * ds["RHO"]
WRHO = WRHO.isel(bottom_top = z_idx).mean(dim="west_east").rename("WRHO")

W_mean   = W.isel(bottom_top=z_idx).mean(dim="west_east")
RHO_mean = ds["RHO"].isel(bottom_top = z_idx).mean(dim="west_east")

WRHO_mean = (W_mean * RHO_mean).rename("WRHO_mean")
WRHO_mix = (WRHO - WRHO_mean).rename("WRHO_mix")

data = xr.merge([
    TOA,
    QOA,
    TO, TA,
    QO, QA,
    WRHO,
    WRHO_mix,
    WRHO_mean,
    ds["PBLH"],
]).mean(dim="west_east")

print("DATA")
print(data)


plot_infos = dict(

    TO = dict(
        factor = 1,
        label = "$T_O$",
        unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
        ylim = [0, 8],
    ),

    TA = dict(
        factor = 1,
        label = "$T_A$",
        unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
        ylim = [0, 8],
    ),

    TOA = dict(
        factor = 1,
        label = "$T_{OA}$",
        unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
        ylim = [0, 8],
    ),

    QO = dict(
        factor = 1,
        label = "$Q_O$",
        unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
        ylim = [0, 8],
    ),

    QA = dict(
        factor = 1,
        label = "$Q_A$",
        unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
        ylim = [0, 8],
    ),

    QOA = dict(
        factor = 1,
        label = "$Q_{OA}$",
        unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
        ylim = [0, 8],
    ),

    WRHO_mean = dict(
        factor = 1,
        label = "$\\overline{\\rho} \\overline{w}$",
        unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
        ylim = [0, 8],
    ),

    WRHO_mix = dict(
        factor = 1,
        label = "$\\overline{ \\rho' w' }$",
        unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
        ylim = [0, 8],
    ),

    PBLH = dict(
        factor = 1,
        label = "$H_\\mathrm{PBL}$",
        unit = "$ \\mathrm{m} $",
        ylim = [0, 1500],
    ),

)


plot_varnames = [
#    ["WRHO_mean", "WRHO_mix"],
    ["PBLH"],
    ["QO", "QA"], ["QOA"],
    ["TO", "TA"], ["TOA"],
]

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


for i, varnames in enumerate(plot_varnames):
    
    show_legend = False
    if len(varnames) > 1:
        show_legend = True

    _ax = ax.flatten()[i]

    for varname in varnames:
        print("Plotting variable: ", varname)
        vardata = data[varname]
        plot_info = plot_infos[varname]

        factor = plot_info["factor"] if "factor" in plot_info else 1.0
        ylim   = plot_info["ylim"] if "ylim" in plot_info else [None, None]

        _ax.plot(t_rel, vardata, label=plot_info["label"])

    #_ax.set_ylim(ylim)

    """
    _ax.set_title("({thumbnail_numbering:s}) {label:s} [{unit:s}]".format(
        thumbnail_numbering = args.thumbnail_numbering[i],
        label = plot_info["label"],
        unit  = plot_info["unit"]),
    )
    """

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

