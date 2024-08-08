import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import wrf_load_helper 
import datetime

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-file', type=str, help='Input file.', required=True)
parser.add_argument('--output', type=str, help='Output filename in png.', default="")
parser.add_argument('--title', type=str, help='Title', default="")
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--varying-param', type=str, help='Parameters. The first and second parameters will be the varying parameters while the rest stays fixed.', required=True, choices=["dSST", "Ug", "Lx"])
parser.add_argument('--fixed-params', type=str, nargs='*', help='Parameters that stay fixed.', required=True, choices=["dSST", "Ug", "Lx"])
parser.add_argument('--thumbnail-numbering', type=str, help='Thumbnail numbering.', default="abcdefghijklmn")
parser.add_argument('--fixed-param-values', type=float, nargs="*", help='The values of the fixed parameters', default=[])
parser.add_argument('--ref-exp-order', type=int, help='The reference case (start from 0) to perform decomposition', default=0)
parser.add_argument('--LH-rng', type=float, nargs=2, help='The values of the LH range', default=[None, None])
parser.add_argument('--HFX-rng', type=float, nargs=2, help='The values of the HFX range', default=[None, None])
parser.add_argument('--spacing', type=float, help='The small separation between different variables', default=0.02)



args = parser.parse_args()

print(args)

ncols = 2
nrows = 1

sel_dict = {}
for i, param in enumerate(args.fixed_params):
    sel_dict[param] = args.fixed_param_values[i]

    if param == "dSST":
        sel_dict[param] /= 1e2

print("sel_dict = ", str(sel_dict))
 

print("Start loading data.")
ds = xr.open_dataset(args.input_file)#, engine="scipy")
ds = ds.sel(**sel_dict)
print(ds)

coord_x = ds.coords[args.varying_param]


plot_infos = dict(


    C_Q_WND_QOA_cx = dict(
        factor = 2.5e6,
        label = "$L_Q \\, \\overline{ C'_Q \\, U'_A \\, Q'_{OA} }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        
    ),

    WND_QOA_cx_mul_C_Q = dict(
        factor = 2.5e6,
        label = "$L_Q \\, \\overline{C}_Q \\, \\overline{ U'_A Q'_{OA} }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        bracket = True,
    ),



    C_Q_QOA_cx_mul_WND = dict(
        factor = 2.5e6,
        label = "$L_Q \\, \\overline{U}_A \\, \\overline{ C_Q' Q'_{OA} }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        
    ),


    C_Q_WND_cx_mul_QOA = dict(
        factor = 2.5e6,
        label = "$L_Q \\, \\overline{Q}_{OA} \\, \\overline{ C_Q' U'_A }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        
    ),

    C_Q_WND_QOA = dict(
        factor = 2.5e6,
        label = "$L_Q \\overline{C}_Q \\, \\overline{U}_A \\, \\overline{Q}_{OA}$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        
    ),

    C_H_WND_TOA = dict(
        factor = 1.0,
        label = "$\\overline{C}_H \\, \\overline{U}_A \\, \\overline{T}_{OA}$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        
    ),


    WND_TOA_cx_mul_C_H = dict(
        label = "$\\overline{C}_H \\, \\overline{ U'_A T'_{OA} }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        
    ),


    C_H_WND_TOA_cx = dict(
        label = "$\\overline{ C'_H \\, U'_A \\, T'_{OA} }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        #
        
    ),

    C_H_TOA_cx_mul_WND = dict(
        label = "$\\overline{U}_A \\, \\overline{ C_H' T'_{OA} }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        #
        
    ),


    C_H_WND_cx_mul_TOA = dict(
        label = "$\\overline{T}_{OA} \\, \\overline{ C_H' U'_A }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        #
        
    ),


    PRECIP = dict(
        factor = 86400.0,
        label = "Precip",
        unit = "$ \\mathrm{mm} / \\mathrm{day} $",
        
    ),


    TO = dict(
        factor = 1,
        label = "$T_O$",
        unit = "$ \\mathrm{K} $",
        
    ),

    TA = dict(
        factor = 1,
        offset = 273.15,
        label = "$\\overline{T_A}$",
        unit = "$ \\mathrm{K} $",
        
    ),

    TOA_m = dict(
        factor = 1,
        label = "$T_{OA}$",
        unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
        
    ),

    QO = dict(
        factor = 1e3,
        label = "$Q_O$",
        unit = "$ \\mathrm{g} / \\mathrm{kg} $",
        
    ),

    QA = dict(
        factor = 1e3,
        label = "$Q_A$",
        unit = "$ \\mathrm{g} / \\mathrm{kg} $",
        
    ),

    QOA_m = dict(
        factor = 1,
        label = "$Q_{OA}$",
        unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
        
    ),

    PBLH = dict(
        factor = 1,
        label = "$\\overline{H_\\mathrm{PBL}}$",
        unit = "$ \\mathrm{m} $",
        
    ),

    HFX = dict(
        factor = 1,
        label = "$\\overline{F}_\\mathrm{sen}$",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
        bracket = False,
    ),

    LH = dict(
        factor = 1,
        label = "$\\overline{F}_\\mathrm{lat}$",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
        bracket = False,
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
        
        #
    ),

    WND_m = dict(
        label = "$\\overline{U} $",
        unit = "$ \\mathrm{m} / \\mathrm{s} $",
        
    ),

    C_H_m = dict(
        label = "$\\overline{C_H} $",
        unit = "$ \\mathrm{m} / \\mathrm{s} $",
    ),

)



# =================================================================
# Figure: HFX decomposition
# =================================================================


# =================================================================
print("Loading matplotlib...")
import matplotlib
if args.no_display:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
print("Done")


print("Plotting decomposition...")


figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = 6,
    h = 4,
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


ax_flattened = ax.flatten()


for k, heatflx in enumerate(["Sensible", "Latent"]):

    _ax = ax_flattened[k]

    varnames = dict(
        Sensible = ["HFX", "C_H_WND_TOA", "WND_TOA_cx_mul_C_H", "C_H_TOA_cx_mul_WND", "C_H_WND_cx_mul_TOA"],
        Latent   = ["LH",  "C_Q_WND_QOA", "WND_QOA_cx_mul_C_Q", "C_Q_QOA_cx_mul_WND", "C_Q_WND_cx_mul_QOA"],
    )[heatflx]

    if args.varying_param == "dSST":
        title_param = "$\\Delta \\mathrm{SST}$"
    elif args.varying_param == "Ug":
        title_param = "$U_g$"
    elif args.varying_param == "Lx":
        title_param = "$L$"
 
    _ax.set_title(
        "(%s) %s heat flux as a function of %s" % (
            args.thumbnail_numbering[k],
            heatflx,
            title_param,
        )
    )
    for i, varname in enumerate(varnames):


        color, linestyle = [
            ("black",      "solid"),
            ("red",        "solid"),
            ("dodgerblue", "dashed"),
            ("magenta",    "dotted"),
            ("green",      "dashdot"),
        ][i]

        

        _plot_info = plot_infos[varname]
        
        factor  = _plot_info["factor"] if "factor" in _plot_info else 1.0
        offset  = _plot_info["offset"] if "offset" in _plot_info else 0.0
        bracket = _plot_info["bracket"] if "bracket" in _plot_info else True

        _plot_data = (ds[varname] + offset) * factor


        _ref_m = _plot_data.sel(stat="mean")[0].to_numpy()
        
        d_m = _plot_data.sel(stat="mean") - _ref_m
        d_s = _plot_data.sel(stat="std")

        if args.varying_param == "dSST":
            _coord_x = coord_x + args.spacing * i
        elif args.varying_param == "Lx":    
            _coord_x = coord_x + args.spacing * i
            #pblh = ds["PBLH"].sel(stat='mean')
            #_coord_x = ( pblh.to_numpy()**2 / (coord_x.to_numpy() * 1e3) )
            #_coord_x = 1 / coord_x.to_numpy()

            #print("COORD_X = ", _coord_x)

        if bracket:
            var_label = "$\\delta ($%s$)$" % (_plot_info["label"],)
        else:    
            var_label = "$\\delta$%s" % (_plot_info["label"],)

        label = "%s (%.2f %s)" % (var_label, _ref_m, _plot_info["unit"])
        _ax.errorbar(_coord_x, d_m, yerr=d_s, fmt='o-', markersize=6, capsize=5, color=color, linewidth=1.5, elinewidth=1.5, linestyle=linestyle, label=label)

        """
        for j in range(len(d_s)):

            #_x = np.array([1.0, 1.0]) * _coord_x[j].to_numpy()
            _x = [ _coord_x[j] ] * 2

            _ax.plot(_x, [d_m[j] - d_s[j], d_m[j] + d_s[j]], color="gray", linestyle="solid")

 
        _ax.scatter(_coord_x, d_m, s=20, c=color)

        _ax.plot(_coord_x, d_m, label=label, color=color, linestyle=linestyle)

        """

        _ax.set_ylabel("[ %s ] " % (_plot_info["unit"]))

        _ax.grid(visible=True)



    if args.varying_param == "dSST":
        _ax.set_xlabel("$\\Delta \\mathrm{SST}$ [ $\\mathrm{K}$ ]")
    elif args.varying_param == "Ug":
        _ax.set_xlabel("$U_\\mathrm{g}$ [ $\\mathrm{m} \\, / \\, \\mathrm{s}$ ]")
    elif args.varying_param == "Lx":
        _ax.set_xlabel("$ L $ [ $\\mathrm{km} $ ]")
        #_ax.set_xlabel("$ H^2_{\\mathrm{pbl}} / L $ [ $\\mathrm{m} $ ]")
        #_ax.set_xlabel("$ L^{-1} $ [ $\\mathrm{km}^{-1} $ ]")


    if heatflx == "Sensible":
        _ax.set_ylim(args.HFX_rng)
    elif heatflx == "Latent":
        _ax.set_ylim(args.LH_rng)



    _ax.legend()




if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()



