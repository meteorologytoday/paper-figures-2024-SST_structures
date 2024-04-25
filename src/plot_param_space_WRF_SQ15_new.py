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

parser.add_argument('--input-file-SQ15', type=str, help='Input directory that contains all cases', required=True)
parser.add_argument('--input-file-WRF', type=str, help='Input directory that contains all cases', required=True)
parser.add_argument('--fixed-Ug', type=float, help='The values of the fixed parameters', required=True)
parser.add_argument('--fixed-RH', type=float, help='The values of the fixed parameters', required=True)
parser.add_argument('--fixed-DTheta', type=float, help='The values of the fixed parameters', required=True)

parser.add_argument('--param1-rng', type=float, nargs=2, help='Parameter 1 range in plotting', default=[None, None])
parser.add_argument('--param2-rng', type=float, nargs=2, help='Parameter 1 range in plotting', default=[None, None])

parser.add_argument('--title', type=str, help='Title', default="")
parser.add_argument('--output', type=str, help='Output file', default="")
parser.add_argument('--no-display', action="store_true")

args = parser.parse_args()
print(args)

pathlist = []
sorting = []

sel_dict_SQ15 = {
    "DTheta" : args.fixed_DTheta,
    "Ug"     : args.fixed_Ug,
    "RH"     : args.fixed_RH,
}

sel_dict_WRF = {
    "Ug"     : args.fixed_Ug,
}


print("sel_dict_SQ15 = ", str(sel_dict_SQ15))
print("sel_dict_WRF = ", str(sel_dict_WRF))
    
coord_x_varname = "Lx"
coord_y_varname = "dSST"

print("Loading input file: ", args.input_file_SQ15)
data_SQ15 = xr.open_dataset(args.input_file_SQ15).sel(**sel_dict_SQ15)


print("Loading input file: ", args.input_file_WRF)
data_WRF = xr.open_dataset(args.input_file_WRF).sel(**sel_dict_WRF)

data_WRF = xr.merge([
    data_WRF["WND_TOA_cx"].rename("UTOA"),
    data_WRF["WND_QOA_cx"].rename("UQOA"),
])


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

print(data_SQ15)
print(data_WRF)


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
        val   = sel_dict_SQ15[k],
    )
    for k, _ in sel_dict_SQ15.items()
]


fig.suptitle(", ".join(pieces))
 
for i, varname in enumerate(varnames):
    
    _ax = ax.flatten()[i]

    plot_info = plot_infos[varname]
    
    coord_x = data_WRF.coords[coord_x_varname].to_numpy().copy()
    coord_y = data_WRF.coords[coord_y_varname].to_numpy().copy()
    coord_info_x = coord_infos[coord_x_varname]
    coord_info_y = coord_infos[coord_y_varname]


    if 'factor' in coord_info_x:
        coord_x *= coord_info_x['factor']

    if 'factor' in coord_info_y:
        coord_y *= coord_info_y['factor']



    for k, (data, color, linestyle) in {
        'SQ15' : (data_SQ15, "k", "-"),
        'WRF'  : (data_WRF,  "r", "-"),
    }.items():

        print("Plot data: ", k)
        _plot_data = data[varname].transpose(coord_x_varname, coord_y_varname).to_numpy() / plot_info['factor']
        _cntr_levs = plot_info['cntr_levs']

        cs = _ax.contour(coord_x, coord_y, np.transpose(_plot_data), _cntr_levs, colors=color)
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


