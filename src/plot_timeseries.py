import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import datetime
import wrf_load_helper 
import os
import wrf_preprocess

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-dirs', nargs="+", type=str, help='Input directories.', required=True)
parser.add_argument('--input-dirs-base', nargs="*", type=str, help='Input directories for baselines. They can be empty.', default=None)
parser.add_argument('--linestyles', nargs="+", type=str, help='Line styles.', default=None)
parser.add_argument('--linecolors', nargs="+", type=str, help='Line styles.', required=True)
parser.add_argument('--labels', nargs="+", type=str, help='Exp names.', default=None)
parser.add_argument('--ncols', type=int, help='Columns of figures.', default=1)
parser.add_argument('--varnames', type=str, nargs="+", help='Variable names.', required=True)
parser.add_argument('--output', type=str, help='Output filename in png.', default="")
parser.add_argument('--extra-title', type=str, help='Title', default="")
parser.add_argument('--smooth', type=int, help='Smooth points. Should be an even number', default=1)
parser.add_argument('--thumbnail-numbering', type=str, help='Thumbnail numbering', default="abcdefghijklmnopqrstu")
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--show-labels', action="store_true")
parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
parser.add_argument('--tick-interval-hour', type=int, help="Ticks interval in hours.", default=12)
parser.add_argument('--time-unit', type=str, help="Unit of time. ", choices=["day", "hour"], default="day")
parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)

args = parser.parse_args()

print(args)


base_exists = args.input_dirs_base is not None
if base_exists:
    if len(args.input_dirs_base) != len(args.input_dirs):
        raise Exception("Error: If `--input-dirs-base` is non-empty, it should have the same number of elements as `--input-dirs`.")


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

def loadData(input_dir):
    
    print("Loading wrf dir: %s" % (input_dir,))
    ds = wrf_load_helper.loadWRFDataFromDir(
        wsm, 
        input_dir,
        beg_time = time_beg,
        end_time = time_end,
        avg=None,
        verbose=False,
        inclusive="left",
    )

    merge_data = [ds, ]   
    merge_data.append(wrf_preprocess.genAnalysis(ds, wsm.data_interval))


    """ 
    if "ACLHF" in ds: 

        if "QFX" in ds:
            QFX = ds["QFX"]
        else:
            ACLQFX = ds["ACLHF"] / 2.5e6
            QFX = (ACLQFX - ACLQFX.shift(time=1)) / wrfout_data_interval.total_seconds()
            QFX = QFX.rename("QFX")
        
        WATER_BUDGET_RES = dWATER_TTLdt - ( QFX - PRECIP )
        WATER_BUDGET_RES = WATER_BUDGET_RES.rename("WATER_BUDGET_RES")
        #QFX = QFX.rename("QFX")

        merge_data.append(WATER_BUDGET_RES)
        merge_data.append(QFX)
    """

    ds = xr.merge(merge_data)

    merge_data = []
    for varname in args.varnames:
        if varname == "BLANK":
            continue
        merge_data.append(ds[varname])

    ds = xr.merge( merge_data )

    for dim in ["west_east", "west_east_stag",]:
        if "west_east" in ds.dims:
            ds = ds.mean(dim=dim)

    ds_computed = ds.rolling(time=args.smooth, center=True).mean().compute()

    return ds_computed
   
    

data = []
for i, input_dir in enumerate(args.input_dirs):
        
    print("[%d] Loading ... " % (i,))

    ds = loadData(input_dir)
    
    if base_exists:
    
        print("[%d] Loading base... " % (i,)) 
        ds_base = loadData(args.input_dirs_base[i])
        

        print("[%d] Take difference... " % (i,))
        ds = ds - ds_base


    data.append(ds)



t = data[0].coords["time"].to_numpy()
t_rel = relTimeInHrs(t) + wrfout_data_interval.total_seconds() / 3600.0

HFX_rng = [ -15, 45]
LH_rng  = [ -10, 200 ]

LH_corr_rng = [-0.5, 15]
HFX_corr_rng = [-1.0, 4.5]

HFX_diff_rng = [-20, 20]
HFX_corr_diff_rng = [-1, 6]
LH_diff_rng = [-35, 45]
LH_corr_diff_rng = [-1, 15]

plot_infos = dict(


    CQ_WND_QOA_cx = dict(
        factor = 2.5e6,
        label = "$L_Q \\, \\overline{ C'_Q \\, U'_A \\, Q'_{OA} }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ylim = LH_corr_rng,
        ylim_diff = LH_corr_diff_rng,
    ),

    WND_QOA_cx_mul_CQ = dict(
        factor = 2.5e6,
        label = "$L_Q \\, \\overline{C}_Q \\, \\overline{ U'_A Q'_{OA} }$",
        label_diff = "$\\delta \\left( L_Q \\, \\overline{C}_Q \\, \\overline{ U'_A Q'_{OA} } \\right)$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ylim = LH_corr_rng,
        ylim_diff = LH_corr_diff_rng,
    ),



    CQ_QOA_cx_mul_WND = dict(
        factor = 2.5e6,
        label = "$L_Q \\, \\overline{U}_A \\, \\overline{ CQ' Q'_{OA} }$",
        label_diff = "$\\delta \\left( L_Q \\, \\overline{U}_A \\, \\overline{ CQ' Q'_{OA} }\\right)$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ylim = LH_corr_rng,
        ylim_diff = LH_corr_diff_rng,
    ),


    CQ_WND_cx_mul_QOA = dict(
        factor = 2.5e6,
        label = "$L_Q \\, \\overline{Q}_{OA} \\, \\overline{ CQ' U'_A }$",
        label_diff = "$\\delta \\left( L_Q \\, \\overline{Q}_{OA} \\, \\overline{ CQ' U'_A } \\right)$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ylim = LH_corr_rng,
        ylim_diff = LH_corr_diff_rng,
    ),

    CQ_WND_QOA = dict(
        factor = 2.5e6,
        label = "$L_Q \\overline{C}_Q \\, \\overline{U}_A \\, \\overline{Q}_{OA}$",
        label_diff = "$\\delta \\left( L_Q \\overline{C}_Q \\, \\overline{U}_A \\, \\overline{Q}_{OA} \\right)$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ylim = LH_rng,
        ylim_diff = LH_diff_rng,
    ),

    CH_WND_TOA = dict(
        factor = 1.0,
        label = "$\\overline{C}_H \\, \\overline{U}_A \\, \\overline{T}_{OA}$",
        label_diff = "$\\delta \\left( \\overline{C}_H \\, \\overline{U}_A \\, \\overline{T}_{OA} \\right)$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ylim = HFX_rng,
        ylim_diff = HFX_diff_rng,
    ),


    WND_TOA_cx_mul_CH = dict(
        label = "$\\overline{C}_T \\, \\overline{ U'_A T'_{OA} }$",
        label_diff = "$\\delta \\left( \\overline{C}_T \\, \\overline{ U'_A T'_{OA} } \\right)$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ylim = HFX_corr_rng,
        ylim_diff = HFX_corr_diff_rng,
    ),


    CH_WND_TOA_cx = dict(
        label = "$\\overline{ C'_H \\, U'_A \\, T'_{OA} }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        #ylim = [-0.2 , 3.6,],
        ylim = HFX_corr_rng,
        ylim_diff = HFX_corr_diff_rng,
    ),

    CH_TOA_cx_mul_WND = dict(
        label = "$\\overline{U}_A \\, \\overline{ CH' T'_{OA} }$",
        label_diff = "$\\delta \\left( \\overline{U}_A \\, \\overline{ CH' T'_{OA} } \\right)$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        #ylim = [-0.2 , 3.6,],
        ylim = HFX_corr_rng,
        ylim_diff = HFX_corr_diff_rng,
    ),


    CH_WND_cx_mul_TOA = dict(
        label = "$\\overline{T}_{OA} \\, \\overline{ CH' U'_A }$",
        label_diff = "$\\delta \\left( \\overline{T}_{OA} \\, \\overline{ CH' U'_A } \\right)$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        #ylim = [-0.2 , 3.6,],
        ylim = HFX_corr_rng,
        ylim_diff = HFX_corr_diff_rng,
    ),

    WATER_BUDGET_RES = dict(
        factor = 86400.0,
        label = "Water budget residue",
        unit = "$ \\mathrm{mm} / \\mathrm{day} $",
        ylim_diff = [-5, 5],
    ),

    QFX = dict(
        factor = 86400.0,
        label = "$\\overline{Q}_{\\mathrm{flx}}$",
        unit = "$ \\mathrm{mm} / \\mathrm{day} $",
        ylim = [-1, 20],
        ylim_diff = [-5, 5],
    ),



    PRECIP = dict(
        factor = 86400.0,
        label = "$\\overline{P}$",
        unit = "$ \\mathrm{mm} / \\mathrm{day} $",
        ylim = [-1, 4],
        ylim_diff = [-5, 5],
    ),

    TTL_RAIN = dict(
        factor = 1.0,
        label = "Accumulated Rain",
        unit = "$ \\mathrm{mm} $",
        ylim_diff = [-20, 20],
    ),



    TO = dict(
        factor = 1,
        label = "$\\overline{\\Theta}_O$",
        unit = "$ \\mathrm{K} $",
        ylim = [0, 8],
    ),

    TA = dict(
        factor = 1,
        offset = 273.15,
        label = "$\\overline{\\Theta}_A$",
        unit = "$ \\mathrm{K} $",
        #ylim = [12, 17],
        #ylim = [12, 17],
    ),

    TOA_m = dict(
        factor = 1,
        label = "$T_{OA}$",
        unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
        ylim = [0, 8],
    ),

    QO = dict(
        factor = 1e3,
        label = "$Q_O^*$",
        unit = "$ \\mathrm{g} / \\mathrm{kg} $",
        ylim = [0, 10],
    ),

    QA = dict(
        factor = 1e3,
        label = "$\\overline{Q}_A$",
        unit = "$ \\mathrm{g} / \\mathrm{kg} $",
        ylim = [-1, 12],
    ),

    QOA_m = dict(
        factor = 1,
        label = "$Q_{OA}$",
        unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
        ylim = [0, 8],
    ),

    PBLH = dict(
        factor = 1,
        label = "$\\overline{H}_\\mathrm{PBL}$",
        unit = "$ \\mathrm{m} $",
        ylim = [0, 2500],
    ),

    HFX = dict(
        factor = 1,
        label = "$\\overline{F}_\\mathrm{sen}$",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
        ylim = HFX_rng,
        ylim_diff = HFX_diff_rng,
    ),

    LH = dict(
        factor = 1,
        label = "$\\overline{F}_\\mathrm{lat}$",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
        ylim = LH_rng,
        ylim_diff = LH_diff_rng,
    ),

    HFX_from_FLHC = dict(
        factor = 1,
        label = "$\\overline{F}_\\mathrm{sen}$ from FLHC",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
        ylim = HFX_rng,
        ylim_diff = HFX_diff_rng,
    ),


    LH_from_FLQC = dict(
        factor = 1,
        label = "$\\overline{F}_\\mathrm{lat}$ from FLQC",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
#        ylim = LH_rng,
#        ylim_diff = LH_diff_rng,
    ),


    HFX_res = dict(
        factor = 1,
        label = "residual $\\overline{F_\\mathrm{sen}}$",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
        ylim = [-15, 15],
        ylim_diff = HFX_diff_rng,
    ),

    LH_res = dict(
        factor = 1,
        label = "residual $\\overline{F_\\mathrm{lat}}$",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
        ylim = [-15, 15],
        ylim_diff = HFX_diff_rng,
    ),



    HFX_approx = dict(
        factor = 1,
        label = "$Approx \\overline{F_\\mathrm{sen}}$",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
        ylim = HFX_rng,
        ylim_diff = HFX_diff_rng,
    ),

    LH_approx = dict(
        factor = 1,
        label = "Approx $\\overline{F_\\mathrm{lat}}$",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
        ylim = LH_rng,
        ylim_diff = LH_diff_rng,

        #ylim = [0, None],
    ),

    SWDOWN = dict(
        factor = 1,
        label = "SWDOWN",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
    ),

    OLR = dict(
        factor = 1,
        label = "Outgoing Longwave Radiation",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
    ),



    WND_sfc = dict(
        label = "$\\overline{U}_A $",
        unit = "$ \\mathrm{m} / \\mathrm{s} $",
        #ylim = [10, 20],
    ),

    CH = dict(
        label = "$\\overline{C}_H $",
        unit = "$ \\mathrm{J} \\, / \\, \\mathrm{K} \\, / \\, \\mathrm{m}^3 $",
    ),

    CQ = dict(
        label = "$\\overline{C}_Q $",
        unit = "$ \\mathrm{kg} \\, / \\, \\mathrm{m}^3 $",
    ),



    QVAPOR_TTL = dict(
        factor = 1.0,
        label = "Integrated Water Vapor",
        unit = "$ \\mathrm{mm} \\, / \\, \\mathrm{m}^2 $",
        ylim = [0, 30],
    ),

    QCLOUD_TTL = dict(
        factor = 1.0,
        label = "Total Cloud Water",
        unit = "$ \\mathrm{mm} \\, / \\, \\mathrm{m}^2 $",
        ylim = [0, 30],
    ),

    QRAIN_TTL = dict(
        factor = 1.0,
        label = "Total Rain Water",
        unit = "$ \\mathrm{mm} \\, / \\, \\mathrm{m}^2 $",
        ylim = [0, 30],
    ),

    IVT = dict(
        factor = 1.0,
        label = "Integrated Vapor Transport",
        unit = "$ \\mathrm{kg} \\, / \\, \\mathrm{m} \\, / \\, \\mathrm{s}$",
        ylim = [0, 600],
    ),


    THETA_MEAN = dict(
        factor = 1.0,
        offset = 273.15,
        label = "Mean Potential Temperature",
        unit = "$ {}^\\circ \\mathrm{C} $",
        #ylim = [10, 20],
    ),

    TKE_TTL = dict(
        factor = 1.0,
        label = "Integrated TKE",
        unit = "$ \\mathrm{J} \\, / \\, \\mathrm{m}^2 $",
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
    w = 5,
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
    sharex=False,
)

#time_fmt="%y/%m/%d %Hh"
#fig.suptitle("%sTime: %s ~ %s" % (args.extra_title, time_beg.strftime(time_fmt), time_end.strftime(time_fmt)))

fig.suptitle("%sTime: %d ~ %d hr" % (args.extra_title, relTimeInHrs(time_beg), relTimeInHrs(time_end),))

for k, _ds in enumerate(data):
   
    print("Plotting the %d-th dataset." % (k,)) 
    
    for i, varname in enumerate(plot_varnames):
    
        _ax = ax.flatten()[i]

        if varname == "BLANK":

            continue

        print("Plotting variable: ", varname)
        plot_info = plot_infos[varname]

        factor = plot_info["factor"] if "factor" in plot_info else 1.0
        offset = plot_info["offset"] if "offset" in plot_info else 0.0


        if base_exists:
            ylim   = plot_info["ylim_diff"] if "ylim_diff" in plot_info else None
            label = plot_info["label_diff"] if "label_diff" in plot_info else "$\\delta$%s" % (plot_info["label"],)
            offset = 0.0

        else:
            ylim   = plot_info["ylim"] if "ylim" in plot_info else None
            label = plot_info["label"]


        vardata = (_ds[varname] - offset) * factor
        _ax.plot(t_rel, vardata, label=args.labels[k], color=args.linecolors[k], linestyle=args.linestyles[k])

        _ax.set_title("(%s) %s" % (
            args.thumbnail_numbering[i],
            label,
        ))
        _ax.set_ylabel("[ %s ]" % (plot_info["unit"],))
    
        if ylim is not None:
            _ax.set_ylim(ylim)

"""
for varname in varnames:
    plot_info = plot_infos[varname]
    _ax.set_ylim(ylim)
"""


total_time = relTimeInHrs(time_end) - relTimeInHrs(time_beg)
        

xticks = args.tick_interval_hour * np.arange(np.ceil(total_time / args.tick_interval_hour)+1)


for _ax in ax.flatten():
    #_ax.legend()

    _ax.grid()

    if args.time_unit == "hr":

        _ax.set_xlabel("[ hr ]")
        _ax.set_xticks(xticks)

    elif args.time_unit == "day":

        _ax.set_xlabel("[ day ]")
        _ax.set_xticks(xticks, labels=["%d" % xtick for xtick in xticks/24 ])

    _ax.set_xlim([relTimeInHrs(time_beg), relTimeInHrs(time_end)])

    if args.show_labels:
        _ax.legend()



for i, varname in enumerate(args.varnames):

    _ax = ax.flatten()[i]

    if varname == "BLANK":
        print("BLANK detected. Remove axis.")
        plt.delaxes(_ax)


if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()

print("Program plot_timeseries.py ends.")
