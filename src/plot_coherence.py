import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import wrf_load_helper 
import datetime
import os
import wrf_preprocess

def corr(arrA, arrB):
    
    arrA_anom = arrA - np.mean(arrA)
    arrB_anom = arrB - np.mean(arrB)

    r = np.sum(arrA_anom * arrB_anom) / (np.sum(arrA_anom**2.0)*np.sum(arrB_anom**2.0))**0.5

    return r


plot_infos = dict(

    SST = dict(
        selector = None,
        wrf_varname = "TSK",
        unit = "K",
        label = "SST",
    ), 


    TA = dict(
        selector = dict(bottom_top=0),
        wrf_varname = "T",
        label = "$\\Theta_{A}$",
        unit = "K",
    ), 

    TOA = dict(
        wrf_varname = "TOA",
        label = "$\\Theta_{OA}$",
        unit = "K",
    ), 

    QOA = dict(
        wrf_varname = "QOA",
        label = "$Q_{OA}$",
        unit = "g / kg",
    ), 

    CH = dict(
        wrf_varname = "CH",
        label = "$C_{H}$",
    ), 

    CQ = dict(
        wrf_varname = "CQ",
        label = "$C_{Q}$",
    ), 


    UA = dict(
        selector = dict(bottom_top=0),
        wrf_varname = "U",
        label = "$u_{A}$",
        unit = "$ \\mathrm{m} \\, / \\, \\mathrm{s}$",
    ), 

    VA = dict(
        selector = dict(bottom_top=0),
        wrf_varname = "V",
        label = "$v_{A}$",
        unit = "$ \\mathrm{m} \\, / \\, \\mathrm{s}$",
    ), 

    VORA = dict(
        selector = dict(bottom_top=0),
        wrf_varname = "VOR",
        label = "$\\zeta_{A}$",
        unit = "$\\mathrm{s}^{-1}$",
    ), 

    DIVA = dict(
        selector = dict(bottom_top=0),
        wrf_varname = "DIV",
        label = "$D_{A}$",
        unit = "$\\mathrm{s}^{-1}$",
    ), 





)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-files', type=str, nargs="+", help='Input directories.', required=True)
    parser.add_argument('--labels', type=str, nargs="+", help='Input directories.', default=None)
    parser.add_argument('--output', type=str, help='Output filename in png.', required=True)
    parser.add_argument('--no-display', action="store_true")
    parser.add_argument('--no-thumbnail-numbering', action="store_true")
    parser.add_argument('--no-legend', action="store_true")
    parser.add_argument('--no-title', action="store_true")
    parser.add_argument('--legend-outside', action="store_true")
    parser.add_argument('--ctl-varname', type=str, help="The varname to be the forcing.", required=True)
    parser.add_argument('--varnames', type=str, nargs="+", help="Varnames to do the analysis.", required=True)
    parser.add_argument('--linestyles', type=str, nargs="+", help="Varnames to do the analysis.", required=True)
    parser.add_argument('--linecolors', type=str, nargs="+", help="Varnames to do the analysis.", required=True)
    parser.add_argument('--thumbnail-numbering', type=str, default="abcdefghijklmn")
    parser.add_argument('--thumbnail-skip', type=int, default=0)



    args = parser.parse_args()

    print(args)

    labels = args.labels

    if labels is None:
        
        labels = [ "%d" % i for i in range(len(args.input_files)) ]
        
    elif len(labels) != len(args.input_files):
        raise Exception("Length of `--labels` (%d) does not equal to length of `--input-dirs` (%d). " % (
            len(labels),
            len(args.input_files),
        ))

    if len(args.linestyles) != len(args.input_files):
        raise Exception("Length of `--linestyles` (%d) does not equal to length of `--input-files` (%d). " % (
            len(args.linecolors),
            len(args.input_files),
        ))

    if len(args.linecolors) != len(args.input_files):
        raise Exception("Length of `--linecolors` (%d) does not equal to length of `--input-files` (%d). " % (
            len(args.linecolors),
            len(args.input_files),
        ))


    data = [
        xr.open_dataset(input_file) for input_file in args.input_files
    ]


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
    import cartopy.crs as ccrs
    import tool_fig_config
    import colorblind

    ncol = 2
    nrow = len(args.varnames)

    figsize, gridspec_kw = tool_fig_config.calFigParams(
        w = 5,
        h = 4,
        wspace = 1.2,
        hspace = 1.0,
        w_left = 1.0,
        w_right = 0.2,
        h_bottom = 2.0,
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
        sharex=False,
        squeeze=False,
    )
   

     
    for i, varname in enumerate(args.varnames):
    
        ax_flatten = ax[i, :]
        
        plot_info = plot_infos[varname]


        varname_label = plot_info["label"] if "label" in plot_info else varname
        varname_label = "$\\delta$%s" % (varname_label,)

        
        _ax1 = ax_flatten[0]
        _ax2 = ax_flatten[1]


        for j, _ds in enumerate(data):

            
            # The estimation of errorbar is suggested by Sarah Gille.
            # Formulation is presented by Bendat and Piersol

            _da = _ds[varname]
            
            
            n_d = len(_da.coords["time"])
            coh_C = _da.sel(stat="coherence_C").mean(dim="time").to_numpy()
            coh_Q = _da.sel(stat="coherence_Q").mean(dim="time").to_numpy()
            coh_S1 = _da.sel(stat="coherence_S1").mean(dim="time").to_numpy()
            coh_S2 = _da.sel(stat="coherence_S2").mean(dim="time").to_numpy()

            gamma_square = (coh_C**2 + coh_Q**2) / (coh_S1 * coh_S2)
            phase_diff = np.arctan(coh_Q / coh_C)


            gamma_square_std = (2**0.5) * ( 1 - gamma_square ) /  (gamma_square * n_d)**0.5
            
            alpha = 0.05 # For error estimation. alpha = 0.05 is 95% significance level
            gamma_significance = 1 - alpha**(1 / (n_d - 1))
            
            phase_diff_std = (1.0 - gamma_square)**(0.5) / (gamma_square**0.5 * ((2 * n_d)**0.5) )
 
            linestyle = args.linestyles[j]
            linecolor = colorblind.BW8color[args.linecolors[j]]
    
            _ax1.plot(
                _da.coords["wvlen"] / 1e3,
                gamma_square**0.5,
                marker='o',
                markersize=6,
                linestyle=linestyle,
                color=linecolor,
                label=labels[j],
            )
                
            trans = transforms.blended_transform_factory(_ax1.transAxes, _ax1.transData)
            _ax1.plot([0, 1], [gamma_significance,]*2, "k--", transform=trans)
     
            _ax2.errorbar(
                _da.coords["wvlen"] / 1e3,
                phase_diff * 180/np.pi,
                phase_diff_std * 180/np.pi,
                fmt='o',
                markersize=6,
                capsize=5,
                linestyle=linestyle,
                color=linecolor,
                label=labels[j],
            )
            



        _ax1.set_ylabel("$\\left| \\gamma_{\\delta SST, \\delta u_A} \\left( k \\right) \\right|^2$")
        _ax2.set_ylabel("$\\phi_{\\delta SST, \\delta u_A} \\left( k \\right)$")
        _ax1.set_ylim([0.0, 1.1])
        _ax2.set_ylim([-90.0, 0.0])

        _ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


        thumbnail_titles = [
            "Squared Coherence",
            "Phase Difference",
        ]

    for i, title in enumerate(thumbnail_titles):

        _ax = ax_flatten[i]
        if not args.no_title:
            if args.no_thumbnail_numbering:
                numbering_str = ""
            else: 
                numbering_str = "(%s)" % (args.thumbnail_numbering[args.thumbnail_skip+i+0],)
            
            if title != "":
                title = " %s" % (title,)
             
            _ax.set_title("%s%s" % (numbering_str, title))

    for _ax in ax.flatten():
        _ax.grid()
        #_ax.set_xticks(Lxs, labels=xticklabels)
        _ax.set_xlabel("$L$ [ km ]")

        if not args.no_legend:
            if args.legend_outside:
                _ax.legend(loc="upper center", ncols=4, mode="expand", bbox_to_anchor=(0., -0.25, 1., .102))
            else:
                _ax.legend()
     
    if args.output != "":
        print("Saving output: ", args.output) 
        fig.savefig(args.output, dpi=200)

    if not args.no_display:
        plt.show()


     





        
