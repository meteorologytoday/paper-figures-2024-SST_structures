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

parser.add_argument('--input-file', type=str, help='Input directory that contains all cases', required=True)
parser.add_argument('--varying-params', type=str, nargs=2, help='Parameters. The first and second parameters will be the varying parameters while the rest stays fixed.', required=True, choices=["dSST", "Ug", "Lx", "DTheta", "RH"])
parser.add_argument('--fixed-params', type=str, nargs='*', help='Parameters that stay fixed.', required=True, choices=["dSST", "Ug", "Lx", "DTheta", "RH"])
parser.add_argument('--fixed-param-values', type=float, nargs="*", help='The values of the fixed parameters', default=[])

parser.add_argument('--param1-rng', type=float, nargs=2, help='Parameter 1 range in plotting', default=[None, None])
parser.add_argument('--param2-rng', type=float, nargs=2, help='Parameter 1 range in plotting', default=[None, None])

parser.add_argument('--output', type=str, help='Output file', default="")
parser.add_argument('--no-display', action="store_true")

args = parser.parse_args()
print(args)

pathlist = []
sorting = []

sel_dict = {}
for i, param in enumerate(args.fixed_params):
    sel_dict[param] = args.fixed_param_values[i]

print("sel_dict = ", str(sel_dict))
    
coord_x_varname = args.varying_params[0]
coord_y_varname = args.varying_params[1]

print("Loading input file: ", args.input_file)
data = xr.open_dataset(args.input_file)




plot_infos = dict(

    UTOA = dict(
        label = "$\\overline{U \\, T_\\mathrm{OA}}$",
        unit  = "$\\times 10^{-2} \\, \\mathrm{K} \\, \\mathrm{m} \\, \\mathrm{s}^{-1}$",
        factor = 1e-2,
        cntr_levs = np.arange(-100, 100, 5),
    ),

    UQOA = dict(
        label = "$\\overline{U \\, q_\\mathrm{OA}}$",
        unit  = "$\\times 10^{-4} \\, \\mathrm{kg} \\, \\mathrm{m}^{-2} \\, \\mathrm{s}^{-1}$",
        factor = 1e-4,
        cntr_levs = np.arange(-100, 100, 1),
    )

)

coord_infos = dict(

    dSST = dict(
        label = "$ \\Delta \\mathrm{SST} $",
        unit  = "$ \\mathrm{K} $",
        factor = 1,
    ),

    Lx   = dict(
        label = "$ L_x $",
        unit  = "$\\mathrm{km}$",
    ),

    Ug   = dict(
        label = "$ U_\\mathrm{g} $",
        unit  = "$ \\mathrm{m} / \\mathrm{s} $",
    ),

    RH   = dict(
        label = "RH",
        unit  = "",
    ),

    DTheta   = dict(
        label = "$ \\Delta \\Theta $",
        unit  = "$ \\mathrm{K} $",
    ),


)



# Plot data
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

varnames = ["UTOA", "UQOA"]

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


pieces = [
    "{label:s} = {val:.1f} {unit:s}".format(
        label = coord_infos[k]["label"],
        unit  = coord_infos[k]["unit"],
        val   = sel_dict[k],
    )
    for k in args.fixed_params
]


fig.suptitle(", ".join(pieces))
 
for i, varname in enumerate(varnames):
    
    _ax = ax.flatten()[i]

    plot_info = plot_infos[varname]
    


    coord_x = data.coords[coord_x_varname].to_numpy().copy()
    coord_y = data.coords[coord_y_varname].to_numpy().copy()
    coord_info_x = coord_infos[coord_x_varname]
    coord_info_y = coord_infos[coord_y_varname]


    if 'factor' in coord_info_x:
        coord_x *= coord_info_x['factor']

    if 'factor' in coord_info_y:
        coord_y *= coord_info_y['factor']


    _plot_data = data[varname].sel(**sel_dict).transpose(coord_x_varname, coord_y_varname).to_numpy() / plot_info['factor']
    _cntr_levs = plot_info['cntr_levs']
    cs = _ax.contour(coord_x, coord_y, np.transpose(_plot_data), _cntr_levs, colors='black')
    plt.clabel(cs)

    _ax.set_title("{label:s} [{unit:s}]".format(
        label = plot_info["label"],
        unit  = plot_info["unit"],
    ))

    _ax.set_xlabel("{label:s} [ {unit:s} ]".format(label=coord_info_x["label"], unit=coord_info_x["unit"]))
    _ax.set_ylabel("{label:s} [ {unit:s} ]".format(label=coord_info_y["label"], unit=coord_info_y["unit"]))


for _ax in ax.flatten():
    _ax.grid()

if not args.no_display:
    plt.show()

if args.output != "":

    print("Saving output: ", args.output)    
    fig.savefig(args.output, dpi=200)


