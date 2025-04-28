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

    ncol = 1
    nrow = 1

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
   
    _ax = ax.flatten()[0]


    for j, _ds in enumerate(data):
        _ds = _ds.isel(Z=0)
                
        Ro     = _ds["Ro"].mean(dim="time").to_numpy()
        Ro_std = _ds["Ro"].mean(dim="time").to_numpy() / _ds.dims["time"]

        linestyle = args.linestyles[j]
        linecolor = colorblind.BW8color[args.linecolors[j]]

        _ax.errorbar(
            _ds.coords["wvlen"] / 1e3,
            Ro,
            Ro_std,
            fmt='o',
            markersize=6,
            capsize=5,
            linestyle=linestyle,
            color=linecolor,
            label=labels[j],
        )
       
    _ax.set_title("Rossby Number")
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


     





        
