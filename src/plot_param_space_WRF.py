import numpy as np
import xarray as xr
import os

import traceback
from pathlib import Path

import argparse



parser = argparse.ArgumentParser(
                    prog = 'plot_skill',
                    description = 'Plot prediction skill of GFS on AR.',
)

parser.add_argument('--title', type=str, help='Title', default=None)
parser.add_argument('--input', type=str, help='Input filename', required=True)
parser.add_argument('--selected-dSST', type=float, help='Selected fixed dSST', required=True)
parser.add_argument('--Ug-rng', type=float, nargs=2, help='Ug range in plotting', default=[None, None])
parser.add_argument('--Lx-rng', type=float, nargs=2, help='Lx range in plotting', default=[None, None])
parser.add_argument('--output', type=str, help='Output file', default="")
parser.add_argument('--no-display', action="store_true")

args = parser.parse_args()
print(args)

pathlist = []
sorting = []

varnames = ["WND_TOA_cx", "WND_QOA_cx", ]


data = xr.open_dataset(args.input)





# Plot data

plot_infos = dict(

    WND_TOA_cx = dict(
        label = "$\\overline{U \\, T_\\mathrm{OA}}$",
        unit  = "$\\times 10^{-2} \\, \\mathrm{K} \\, \\mathrm{m} \\, \\mathrm{s}^{-1}$",
        factor = 1e-2,
        cntr_levs = np.arange(-100, 100, 5),
    ),

    WND_QOA_cx = dict(
        label = "$\\overline{U \\, q_\\mathrm{OA}}$",
        unit  = "$\\times 10^{-4} \\, \\mathrm{kg} \\, \\mathrm{m}^{-2} \\, \\mathrm{s}^{-1}$",
        factor = 1e-4,
        cntr_levs = np.arange(-100, 100, 1),
    )

)



print("Loading Matplotlib...")
import matplotlib as mpl
if args.no_display is False:
    mpl.use('TkAgg')
else:
    mpl.use('Agg')
    mpl.rc('font', size=15)
    mpl.rc('axes', labelsize=15)

print("Done.")     
 
 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mticker
import tool_fig_config


ncol = len(varnames)
nrow = 1

figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = 6,
    h = 6,
    wspace = 1.0,
    hspace = 1.0,
    w_left = 1.0,
    w_right = 2.0,
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


for i, varname in enumerate(varnames):    
    
    _ax = ax.flatten()[i]

    plot_info = plot_infos[varname]
    _plot_data = data[varname].sel(dSST=args.selected_dSST).to_numpy()
    print("%s : mean = %f , std = %f" % (varname, _plot_data.mean(), _plot_data.std()))
    _plot_data /= plot_info["factor"]

    if 'cntr_levs' in plot_info:
        cntr_levs = plot_info['cntr_levs']
    else:
        cntr_levs = 10 



    cs = _ax.contour(data.coords['Lx'], data.coords['Ug'], _plot_data, cntr_levs, colors='black')



    plt.clabel(cs)
    _ax.set_title("{label:s} [{unit:s}]".format(
        label = plot_info["label"],
        unit  = plot_info["unit"],
    ))
    _ax.set_xlabel("$ L_x $ [ km ]")
    _ax.set_ylabel("$ U_\\mathrm{g} $ [ m / s ]")
    
    
    _ax.grid()
    _ax.set_ylim(args.Ug_rng)
    _ax.set_xlim(args.Lx_rng)

if args.title is not None:
    fig.suptitle(args.title)


if not args.no_display:
    plt.show()

if args.output != "":

    print("Saving output: ", args.output)    
    fig.savefig(args.output, dpi=200)


