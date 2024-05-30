import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import datetime
import wrf_load_helper 
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-dirs', nargs="+", type=str, help='Input directories.', required=True)
parser.add_argument('--ncols', type=int, help='Columns of figures.', default=1)
parser.add_argument('--varnames', type=str, nargs="+", help='Variable names.', required=True)
parser.add_argument('--output', type=str, help='Output filename in png.', default="")
parser.add_argument('--extra-title', type=str, help='Title', default="")
parser.add_argument('--thumbnail-numbering', type=str, help='Thumbnail numbering', default="abcdefghijklmn")
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
parser.add_argument('--tick-interval-hour', type=int, help="Ticks interval in hours.", default=12)
parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)

args = parser.parse_args()

print(args)


Nvars = len(args.varnames)
ncols = args.ncols
nrows = int(np.ceil( Nvars / ncols))




exp_beg_time = pd.Timestamp(args.exp_beg_time)
wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)
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
data = []
for i, input_dir in enumerate(args.input_dirs):

    print("Loading wrf dir: %s" % (input_dir,))
    ds = wrf_load_helper.loadWRFDataFromDir(
        wsm, 
        input_dir,
        beg_time = time_beg,
        end_time = time_end,
        prefix="analysis_",
        suffix=".nc",
        avg=None,
        verbose=False,
        inclusive="left",
    )

    TTL_RAIN = ds["RAINNC"] + ds["RAINC"]
    PRECIP = ( TTL_RAIN - TTL_RAIN.shift(time=1) ) / wrfout_data_interval.total_seconds()
    PRECIP = PRECIP.rename("PRECIP") 
    
    ds = xr.merge([ds, PRECIP])
    
    ds = xr.merge( [ ds[varname] for varname in args.varnames ] ).load()

    data.append(ds)


t = data[0].coords["time"].to_numpy()
t_rel = relTimeInHrs(t) + wrfout_data_interval.total_seconds() / 3600

plot_infos = dict(


    C_Q_WND_QOA_cx = dict(
        factor = 2.5e6,
        label = "$L_q \\, \\overline{ C'_H \\, U' \\, Q'_{OA} }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        #ylim = [0, 1],
    ),

    WND_QOA_cx_mul_C_Q = dict(
        factor = 2.5e6,
        label = "$L_q \\, \\overline{C}_Q \\, \\overline{ U' Q'_{OA} }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        #ylim = [0, 1],
    ),



    C_Q_QOA_cx_mul_WND = dict(
        factor = 2.5e6,
        label = "$L_q \\, \\overline{U} \\, \\overline{ C_Q' Q'_{OA} }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        #ylim = [0, 1],
    ),


    C_Q_WND_cx_mul_QOA = dict(
        factor = 2.5e6,
        label = "$L_q \\, \\overline{Q}_{OA} \\, \\overline{ C_Q' U' }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        #ylim = [0, 1],
    ),

    C_Q_WND_QOA = dict(
        factor = 2.5e6,
        label = "$L_q \\overline{C}_Q \\, \\overline{U} \\, \\overline{Q}_{OA}$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        #ylim = [0, 1],
    ),

    C_H_WND_TOA = dict(
        factor = 1.0,
        label = "$\\overline{C}_H \\, \\overline{U} \\, \\overline{T}_{OA}$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        #ylim = [0, 1],
    ),


    WND_TOA_cx_mul_C_H = dict(
        label = "$\\overline{C}_T \\, \\overline{ U' T'_{OA} }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        #ylim = [0, 1],
    ),



    PRECIP = dict(
        factor = 86400.0,
        label = "Precip",
        unit = "$ \\mathrm{mm} / \\mathrm{day} $",
        #ylim = [0, 1],
    ),


    TO = dict(
        factor = 1,
        label = "$T_O$",
        unit = "$ \\mathrm{K} $",
        ylim = [0, 8],
    ),

    TA = dict(
        factor = 1,
        offset = 273.15,
        label = "$\\overline{T_A}$",
        unit = "$ \\mathrm{K} $",
        #ylim = [13, 17],
    ),

    TOA_m = dict(
        factor = 1,
        label = "$T_{OA}$",
        unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
        ylim = [0, 8],
    ),

    QO = dict(
        factor = 1e3,
        label = "$Q_O$",
        unit = "$ \\mathrm{g} / \\mathrm{kg} $",
        ylim = [0, 10],
    ),

    QA = dict(
        factor = 1e3,
        label = "$Q_A$",
        unit = "$ \\mathrm{g} / \\mathrm{kg} $",
        #ylim = [0, 10],
    ),

    QOA_m = dict(
        factor = 1,
        label = "$Q_{OA}$",
        unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
        ylim = [0, 8],
    ),

    PBLH = dict(
        factor = 1,
        label = "$\\overline{H_\\mathrm{PBL}}$",
        unit = "$ \\mathrm{m} $",
        ylim = [0, 1500],
    ),

    HFX = dict(
        factor = 1,
        label = "$\\overline{F_\\mathrm{sen}}$",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",

    ),

    LH = dict(
        factor = 1,
        label = "$\\overline{F_\\mathrm{lat}}$",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
        ylim = [0, 100],
        #ylim = [0, None],
    ),

 
    HFX_approx = dict(
        factor = 1,
        label = "$Approx \\overline{F_\\mathrm{sen}}$",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
    ),

    LH_approx = dict(
        factor = 1,
        label = "Approx $\\overline{F_\\mathrm{lat}}$",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
        ylim = [0, 100],
        #ylim = [0, None],
    ),

    H_inf = dict(
        factor = 1,
        label = "$\\overline{H_{\\infty}}$",
        unit = "$ \\mathrm{m} $",
    ),

    L_MO = dict(
        factor = 1,
        label = "$\\overline{L_{*}}$",
        unit = "$ \\mathrm{m} $",
    ),




)


plot_varnames = args.varnames

print("Loading matplotlib...")
import matplotlib
if args.no_display:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
print("Done")

figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = 4,
    h = 3,
    wspace = 1.0,
    hspace = 0.7,
    w_left = 1.0,
    w_right = 1.0,
    h_bottom = 1.0,
    h_top = 1.0,
    ncol = ncols,
    nrow = nrows,
)


fig, ax = plt.subplots(
    nrows, ncols,
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

for k, _ds in enumerate(data):
   
    print("Plotting the %d-th dataset." % (k,)) 
    
    for i, varname in enumerate(plot_varnames):
        
        _ax = ax.flatten()[i]

        print("Plotting variable: ", varname)
        plot_info = plot_infos[varname]

        factor = plot_info["factor"] if "factor" in plot_info else 1.0
        offset = plot_info["offset"] if "offset" in plot_info else 0.0
        ylim   = plot_info["ylim"] if "ylim" in plot_info else None


        vardata = (_ds[varname] - offset) * factor
        _ax.plot(t_rel, vardata, label=plot_info["label"], color=args.linecolors[k], linestyle=args.linestyles[k])

        _ax.set_title("(%s)" % (args.thumbnail_numbering[i],))
        _ax.set_ylabel("%s [ %s ]" % (plot_info["label"], plot_info["unit"]))
    
        if ylim is not None:
            _ax.set_ylim(ylim)

"""
for varname in varnames:
    plot_info = plot_infos[varname]
    _ax.set_ylim(ylim)
"""


total_time = relTimeInHrs(time_end) - relTimeInHrs(time_beg)

for _ax in ax.flatten():
    #_ax.legend()
    _ax.grid()
    _ax.set_xlabel("[ hr ]")
    
    #_ax.set_xticks(args.tick_interval_hour * np.arange(np.ceil(total_time / args.tick_interval_hour)+1))

for _ax in ax.flatten()[Nvars:]:
    fig.delaxes(_ax)


if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()

