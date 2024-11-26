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



)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-dirs', type=str, nargs="+", help='Input directories.', required=True)
    parser.add_argument('--input-dirs-base', type=str, nargs="+", help='Input directories.', required=True)
    parser.add_argument('--tracking-wnms', type=int, nargs="+", help='The wave number to trace.', required=True)
    parser.add_argument('--labels', type=str, nargs="+", help='Input directories.', default=None)
    parser.add_argument('--output', type=str, help='Output filename in png.', required=True)
    parser.add_argument('--no-display', action="store_true")
    parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
    parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
    parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
    parser.add_argument('--number-of-harmonics', type=int, help='Number of frames in each wrfout file.', default=None)
    parser.add_argument('--ctl-varname', type=str, help="The varname to be the forcing.", required=True)
    parser.add_argument('--varnames', type=str, nargs="+", help="Varnames to do the linar response.", required=True)
    parser.add_argument('--linestyles', type=str, nargs="+", help="Varnames to do the analysis.", required=True)
    parser.add_argument('--linecolors', type=str, nargs="+", help="Varnames to do the analysis.", required=True)
    parser.add_argument('--thumbnail-numbering', type=str, default="abcdefghijklmn")
    parser.add_argument('--labeled-wvlen', type=int, nargs="*", help='Number of frames in each wrfout file.', default=[])
    parser.add_argument('--wrfout-suffix', type=str, default="")
    parser.add_argument('--magnitude-threshold', type=float, help='The threshold that set direction=0 if magnitude is too low.', default=1e-5)



    args = parser.parse_args()

    print(args)

    labels = args.labels

    if labels is None:
        
        labels = [ "%d" % i for i in range(len(args.input_dirs)) ]
        
    elif len(labels) < len(args.input_dirs):
        raise Exception("Length of `--labels` (%d) does not equal or exceed to length of `--input-dirs` (%d). " % (
            len(labels),
            len(args.input_dirs),
        ))

    if len(args.linestyles) < len(args.input_dirs):
        raise Exception("Length of `--linestyles` (%d) does not equal or exceed to length of `--input-dirs` (%d). " % (
            len(args.linestyles),
            len(args.input_dirs),
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
        
        d_anom = dict()

        for varname in args.varnames + [args.ctl_varname,]:

            if varname in d_anom:
                print("Varname %s is already loaded. Skip." % (varname,))
                continue

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
            d_anom[varname] = dvar



           
        X0 = d_anom[args.ctl_varname]
        sp_X0 = np.fft.fft(X0) / Nx
        d = dict()
        for varname in args.varnames:
            
            print("Doing coherence analysis of %s" % (varname,)) 
            X1 = d_anom[varname]
            sp_X1 = np.fft.fft(X1) / Nx
            
            G_X0X1 = 2 * np.real(sp_X0 * np.conjugate(sp_X1))
            G_X0X0 = np.abs(sp_X0)**2
            G_X1X1 = np.abs(sp_X1)**2
           

            coherence = G_X0X1 / (G_X0X0 * G_X1X1)
            
            test_signal = G_X0X0 * G_X1X1 
    
            #coherence[test_signal < 1e-5] = np.nan
            print("G_X0X0: ", G_X0X0[:11])
            print("G_X1X1: ", G_X1X1[:11])
            print("G_X0X1: ", G_X0X1[:11])
            print("coherence: ", coherence[:11])
            d[varname] = dict(
                coherence = coherence,
                freq = freq,
            )
 
            

        data.append(d)



    if args.number_of_harmonics is None:
        
        cut_off = freq_N

    else:
        
        cut_off = args.number_of_harmonics

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

    ncol = 1
    nrow = len(args.input_dirs)

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
    
    
    label_wvlen = np.array(args.labeled_wvlen) * 1e3
    label_freq = label_wvlen**(-1) 
    label_wvlen_text = ["%d" % wvlen for wvlen in (label_wvlen / 1e3) ]
    
    for i, _d in enumerate(data):
       
        _ax = ax[i, 0]
        label = args.labels[i]


        for j, varname in enumerate(args.varnames):
            
    
            _dd = _d[varname]
            plot_info = plot_infos[varname]

            linestyle = args.linestyles[i]
            linecolor = args.linecolors[i]

            varname_label = plot_info["label"] if "label" in plot_info else varname
            unit = plot_info["unit"] if "unit" in plot_info else "???"
 
            x = freq[1:cut_off]

            coherence = _dd["coherence"][1:cut_off]
 
            _ax.plot( x, coherence, label=labels[i], marker='o', linestyle=linestyle, color=linecolor)
            


            
        _ax.set_title("(%s) %s" % (args.thumbnail_numbering[i], label))
        _ax.grid()
        _ax.set_xticks(label_freq, labels=label_wvlen_text)
        _ax.set_xlabel("Wavelength [ km ]")
        
        #_ax.set_ylim([-180, 180])
        #_ax.set_yticks(np.arange(-180, 210, 30))
        _ax.set_ylabel("Coherence [ None ]")
                

    if args.output != "":
        print("Saving output: ", args.output) 
        fig.savefig(args.output, dpi=200)

    if not args.no_display:
        plt.show()


     





        
