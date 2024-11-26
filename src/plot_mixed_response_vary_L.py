import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import wrf_load_helper 
import datetime
import os
import wrf_preprocess

plot_infos = dict(

    SST = dict(
        selector = None,
        wrf_varname = "TSK",
        unit = "K",
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
        label = "$D_{A}$ wavelength = $L$",
        unit = "$\\mathrm{s}^{-1}$",
    ), 

    DIVA90PR = dict(
        selector = dict(bottom_top=0),
        wrf_varname = "DIV",
        label = "$D^{\\mathrm{90}}_{A}$",
        unit = "$\\mathrm{s}^{-1}$",
    ), 

    CONA90PR = dict(
        selector = dict(bottom_top=0),
        wrf_varname = "DIV",
        label = "$C^{\\mathrm{90}}_{A}$",
        unit = "$\\mathrm{s}^{-1}$",
    ), 



)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-dirs', type=str, nargs="+", help='Input directories.', required=True)
    parser.add_argument('--input-dirs-base', type=str, nargs="+", help='Input directories.', required=True)
    parser.add_argument('--tracking-wnms', type=int, nargs="+", help='The wave number to trace.', required=True)
    parser.add_argument('--labels', type=str, nargs="+", help='Input directories.', default=None)
    parser.add_argument('--output', type=str, help='Output filename in png.', required=True)
    parser.add_argument('--no-display', action="store_true")
    parser.add_argument('--no-title', action="store_true")
    parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
    parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
    parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
    parser.add_argument('--number-of-harmonics', type=int, help='Number of frames in each wrfout file.', default=None)
    parser.add_argument('--varnames', type=str, nargs="+", help="Varnames to do the analysis.", required=True)
    parser.add_argument('--linestyles', type=str, nargs="+", help="Varnames to do the analysis.", required=True)
    parser.add_argument('--linecolors', type=str, nargs="+", help="Varnames to do the analysis.", required=True)
    parser.add_argument('--thumbnail-numbering', type=str, default="abcdefghijklmn")
    parser.add_argument('--thumbnail-skip', type=int, default=0)
    parser.add_argument('--labeled-wvlen', type=int, nargs="*", help='Number of frames in each wrfout file.', default=[])
    parser.add_argument('--wrfout-suffix', type=str, default="")

    args = parser.parse_args()

    print(args)

    labels = args.labels

    if labels is None:
        
        labels = [ "%d" % i for i in range(len(args.input_dirs)) ]
        
    elif len(labels) != len(args.input_dirs):
        raise Exception("Length of `--labels` (%d) does not equal to length of `--input-dirs` (%d). " % (
            len(labels),
            len(args.input_dirs),
        ))

    if len(args.linestyles) != len(args.varnames):
        raise Exception("Length of `--linestyles` (%d) does not equal to length of `--varnames` (%d). " % (
            len(args.linestyles),
            len(args.varnames),
        ))

    if len(args.linecolors) != len(args.varnames):
        raise Exception("Length of `--linecolors` (%d) does not equal to length of `--varnames` (%d). " % (
            len(args.linecolors),
            len(args.varnames),
        ))


        
    same_base = False
    if len(args.input_dirs_base) == 1:
       
        args.input_dirs_base = [ args.input_dirs_base[0] ] * len(args.input_dirs) 


    if np.all( [ input_dir_base == args.input_dirs_base[0] for input_dir_base in args.input_dirs_base  ] ):
        same_base = True
        print("# same_base = ", same_base)


    if len(args.input_dirs_base) != len(args.input_dirs):
        
        raise Exception("Length of `--input-dirs-base` (%d) does not equal to length of `--input-dirs` (%d). " % (
            len(args.input_dirs_base),
            len(args.input_dirs),
        ))

 
    exp_beg_time = pd.Timestamp(args.exp_beg_time)
    wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)
    time_beg = exp_beg_time + pd.Timedelta(hours=args.time_rng[0])
    time_end = exp_beg_time + pd.Timedelta(hours=args.time_rng[1])

    wsm = wrf_load_helper.WRFSimMetadata(
        start_datetime  = exp_beg_time,
        data_interval   = wrfout_data_interval,
        frames_per_file = args.frames_per_wrfout_file,
    )
    
    # Loading     
                
    data = [] 

    for i in range(len(args.input_dirs)):
        
        input_dir_base = args.input_dirs_base[i] 
        input_dir      = args.input_dirs[i]
         
        print("Loading base wrf dir: %s" % (input_dir_base,))


        if i == 0 or not same_base:
            ds_base = wrf_load_helper.loadWRFDataFromDir(
                wsm, 
                input_dir_base,
                beg_time = time_beg,
                end_time = time_end,
                suffix=args.wrfout_suffix,
                avg=None,
                verbose=False,
                inclusive="left",
            )

            DX = ds_base.attrs["DX"]

            ds_base = xr.merge([
                ds_base,
                wrf_preprocess.genAnalysis(ds_base, wsm.data_interval),
            ]).mean(dim="time")


        Nx = len(ds_base.coords["west_east"])
        X_sU = DX * np.arange(Nx+1)
        X_sT = (X_sU[1:] + X_sU[:-1]) / 2
        freq = np.fft.fftfreq(Nx, d=DX)

        freq_N = Nx // 2
        
         
     
        print("Loading the %d-th wrf dir: %s" % (i, input_dir,))

        ds = wrf_load_helper.loadWRFDataFromDir(
            wsm, 
            input_dir,
            beg_time = time_beg,
            end_time = time_end,
            suffix=args.wrfout_suffix,
            avg=None,
            verbose=False,
            inclusive="left",
        )
            
        ds = xr.merge([
            ds,
            wrf_preprocess.genAnalysis(ds, wsm.data_interval),
        ]).mean(dim="time")
        
        d = dict()
        for varname in args.varnames + ["SST",]:
            plot_info = plot_infos[varname]

            selector = plot_info["selector"] if "selector" in plot_info else None
            wrf_varname = plot_info["wrf_varname"] if "wrf_varname" in plot_info else varname

            da_base = ds_base[wrf_varname]
            da = ds[wrf_varname]


            if selector is not None:
                da_base = da_base.isel(**selector)
                da      = da.isel(**selector)
           
             
            if "south_north" in da.dims:
                da = da.isel(south_north=0)
                da_base = da_base.isel(south_north=0)
            
            elif "south_north_stag" in da.dims:
                da = da.isel(south_north_stag=0)
                da_base = da_base.isel(south_north_stag=0)

            dvar = da - da_base
            dvar = dvar.to_numpy()

            if varname == "UA":
                dvar = ( dvar[1:] + dvar[:-1] ) / 2
 
            # Compute transfer function
            d[varname] = dict(
                sp = np.fft.fft(dvar) / Nx,
                freq = freq,
                Lx = DX * Nx,
                Nx = Nx,
                dvar = dvar,
            )


        data.append(d)

    # Convert data into a function of wvn
    tracking = dict()
    for varname in args.varnames:

        mags = np.zeros((len(data),))
        angs = np.zeros_like(mags)
        Lxs = np.zeros_like(mags)
        tracking_wnms = np.array(args.tracking_wnms)

        if varname in ["DIVA90PR", "CONA90PR"]:

            tmp = np.zeros((len(data),))
            for i, d in enumerate(data):
     
                dd = d[varname]["dvar"]
                tracking_wnm = args.tracking_wnms[i]
                
                if varname == "CONA90PR":
                    dd = - dd


                # Calculate the 90th percentile
                threshold = np.percentile(dd, 90)

                # Get the values greater than or equal to the threshold
                tmp[i] = np.mean(dd[dd >= threshold]) 
                Lxs[i] = d[varname]["Lx"] / tracking_wnm
            
            tracking[varname] = dict(data=tmp, Lxs=Lxs)

        else:
            for i, d in enumerate(data):
     
                dd = d[varname]

                tracking_wnm = args.tracking_wnms[i]
                adj_phase_rad = - 0.5 / (dd["Nx"] / tracking_wnm) * 2 * np.pi
                adj_cpx = np.cos(adj_phase_rad) + 1j * np.sin(adj_phase_rad)
                 

                sp = dd["sp"][tracking_wnm] * adj_cpx
                
                

                mags[i] = np.abs(sp)
                angs[i] = np.angle(sp, deg=True)
                Lxs[i] = d[varname]["Lx"] / tracking_wnm
            
                
            tracking[varname] = dict(mags=mags, angs=angs, Lxs=Lxs) 



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
        w = 6,
        h = 4,
        wspace = 1.0,
        hspace = 1.0,
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
        sharex=False,
        squeeze=False,
    )
    
    for i, varname in enumerate(args.varnames):
        
        _tracking = tracking[varname]
        plot_info = plot_infos[varname]

        linestyle = args.linestyles[i]
        linecolor = colorblind.BW8color[args.linecolors[i]]

        varname_label = plot_info["label"] if "label" in plot_info else varname

        _ax1 = ax[0, 0]
        Lxs = _tracking["Lxs"] / 1e3

        if varname in ["DIVA90PR", "CONA90PR"]:
            
            d = _tracking["data"]
            rel_mags = d / d[0] * 100
            _ax1.plot(Lxs, rel_mags, marker='o', linestyle=linestyle, color=linecolor, label=varname_label)

        else:
            mags = _tracking["mags"]
            angs = _tracking["angs"]

            #wnms = _tracking["wnms"]
            rel_mags = mags / mags[0] * 100

            _ax1.plot(Lxs, rel_mags, marker='o', linestyle=linestyle, color=linecolor, label=varname_label)
            

        if ( not args.no_title ) and i == 0: 
            _ax1.set_title("(%s) Relative magnitude of the linear response" % (args.thumbnail_numbering[args.thumbnail_skip+0],))




    #label_wvlen = np.array(args.labeled_wvlen) * 1e3
    xticks = Lxs
    xticklabels = ["%d" % np.round(Lx) for Lx in (Lxs) ]

    for _ax in ax.flatten():
        _ax.grid()
        _ax.set_xticks(Lxs, labels=xticklabels)
        _ax.set_xlabel("$L$ [ km ]")
        _ax.legend()
 
    for _ax in ax[:, 0].flatten():
        _ax.set_ylabel("[ $\\%$ ]")
        
    if args.output != "":
        print("Saving output: ", args.output) 
        fig.savefig(args.output, dpi=200)

    if not args.no_display:
        plt.show()


     





        
