import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import datetime
import wrf_load_helper
import wrf_preprocess 
import cmocean
from shared_constants import *

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-dir', type=str, help='Input directory.', required=True)
parser.add_argument('--input-dir-base', type=str, help='Input directory for base.', default=None)
parser.add_argument('--output1', type=str, help='Output filename in png.', default="")
parser.add_argument('--output2', type=str, help='Output filename in png.', default="")
parser.add_argument('--extra-title', type=str, help='Title', default="")
parser.add_argument('--overwrite-title', type=str, help='If set then title will be set to this.', default="")

parser.add_argument('--no-display', action="store_true")
parser.add_argument('--plot-check', action="store_true")
parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
#parser.add_argument('--ref-time-rng', type=int, nargs=2, help="Reference time range in hours after --exp-beg-time", required=True)
parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)

parser.add_argument('--part1-z-rng', type=float, nargs=2, help='The plotted height rng in meters.', default=[0, 5000.0])
parser.add_argument('--part2-z-rng', type=float, nargs=2, help='The plotted height rng in meters.', default=[0, 2000.0])
parser.add_argument('--part1-x-rng', type=float, nargs=2, help='The plotted height rng in kilometers', default=[None, None])
parser.add_argument('--part1-U10-rng', type=float, nargs=2, help='The plotted surface wind in m/s', default=[None, None])
parser.add_argument('--part1-U500-rng', type=float, nargs=2, help='The plotted height rng in kilometers', default=[None, None])
parser.add_argument('--part1-DIVVOR10-rng', type=float, nargs=2, help='DIV VOR 10m. In unit of 1e-5 /s', default=[None, None])
parser.add_argument('--part1-DIVVOR500-rng', type=float, nargs=2, help='DIV VOR 500m. In unit of 1e-5 /s', default=[None, None])
parser.add_argument('--part1-PRECIP-rng', type=float, nargs=2, help='PRECIP in unit of mm/day', default=[None, None])
parser.add_argument('--part1-Q-rng', type=float, nargs=2, help='The plotted height rng in kilometers', default=[None, None])
parser.add_argument('--part1-SST-rng', type=float, nargs=2, help='Title', default=[-5, 5])
parser.add_argument('--part1-W-levs', type=float, nargs=3, help='The plotted W contours. Three parameters will be fed into numpy.linspace', default=[-8, 8, 17])
parser.add_argument('--part1-TKE-levs', type=float, nargs=3, help='The plotted W contours. Three parameters will be fed into numpy.linspace', default=[-1, 1, 11])
parser.add_argument('--part1-x-rolling', type=int, help='The plotted height rng in kilometers', default=1)

parser.add_argument('--part2-THETA-rng', type=float, nargs=2, help='Theta range in K', default=[None, None])
parser.add_argument('--part2-Nfreq2-rng', type=float, nargs=2, help='Theta range in 1e-2/s^2', default=[None, None])
parser.add_argument('--part2-TKE-rng', type=float, nargs=2, help='The plotted surface wind in m/s', default=[None, None])
parser.add_argument('--part2-DTKE-rng', type=float, nargs=2, help='The plotted surface wind in m/s', default=[None, None])
parser.add_argument('--part2-U-rng', type=float, nargs=2, help='U range in m/s', default=[None, None])
parser.add_argument('--part2-Q-rng', type=float, nargs=2, help='Q range in g/kg', default=[None, None])

parser.add_argument('--part2-tke-analysis', type=str, help='analysis beg time', choices=["TRUE", "FALSE"], default="FALSE")
parser.add_argument('--thumbnail-skip-part1', type=int, help='Skip of thumbnail numbering.', default=0)
parser.add_argument('--thumbnail-skip-part2', type=int, help='Skip of thumbnail numbering.', default=0)
parser.add_argument('--thumbnail-numbering', type=str, help='Skip of thumbnail numbering.', default="abcdefghijklmn")

parser.add_argument('--plot-part1', action="store_true")
parser.add_argument('--plot-part2', action="store_true")

args = parser.parse_args()

print(args)

base_exists = args.input_dir_base is not None    

if not args.plot_part1 and not args.plot_part2:
    raise Exception("Either one of `--plot-part1` or `--plot-part2` should be activated.")


if args.plot_part1:
    
    if not base_exists:
        raise Exception("If `--plot-part1`, base need to be provided.")
    
    print("Plan to generate plot 1")

if args.plot_part2:
    print("Plan to generate plot 2")




exp_beg_time = pd.Timestamp(args.exp_beg_time)
wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)

time_beg = exp_beg_time + pd.Timedelta(minutes=args.time_rng[0])
time_end = exp_beg_time + pd.Timedelta(minutes=args.time_rng[1])

def relTimeInHrs(t):
    return (t - exp_beg_time.to_datetime64()) / np.timedelta64(1, 'h')


def horDecomp(da, name_m="mean", name_p="prime"):
    m = da.mean(dim="west_east").rename(name_m)
    p = (da - m).rename(name_p) 
    return m, p


# Loading data
print("Loading wrf dir: %s" % (args.input_dir,))






def loadData(input_dir):

    #print("Loading dir: ", input_dir)

    wsm = wrf_load_helper.WRFSimMetadata(
        start_datetime  = exp_beg_time,
        data_interval   = wrfout_data_interval,
        frames_per_file = args.frames_per_wrfout_file,
    )



    ds_nonavg = wrf_load_helper.loadWRFDataFromDir(
        wsm, 
        input_dir,
        beg_time = time_beg,
        end_time = time_end,
        prefix="wrfout_d01_",
        avg = None, #"ALL",
        verbose=False,
        inclusive="both",
    )


   
    ds_extra = wrf_preprocess.genAnalysis(ds_nonavg, wsm.data_interval)
    
    ds_extra = xr.merge( [ds_extra[varname] for varname in ["PRECIP", "TA", "QA", "TO", "QO", "VOR10", "DIV10", "DIV", "VOR", "U_T", "WND"]] ).mean(dim="time")
     
    #ACCRAIN = ds["RAINNC"] + ds["RAINC"]

    #PRECIP = ( ACCRAIN.isel(time=-1) - ACCRAIN.isel(time=0) ) / ( time_end - time_beg - wrfout_data_interval ).total_seconds()


    ds = wrf_load_helper.loadWRFDataFromDir(
        wsm, 
        input_dir,
        beg_time = time_beg,
        end_time = time_end,
        prefix="wrfout_d01_",
        avg = "ALL",
        verbose=False,
        inclusive="both",
    )

    ds = xr.merge([ds, ds_extra])


    WND10 = ((ds.U10**2 + ds.V10**2)**0.5).rename("WND10")
    ds = xr.merge([ds, WND10])

    has_TKE = "QKE" in ds


    # TKE variables
    def W2T(da_W):
        
        da_T = xr.zeros_like(ds["T"])
        
        da_W = da_W.to_numpy()
        da_T[:, :, :, :] = ( da_W[:, :-1, :, :] + da_W[:, 1:, :, :] ) / 2.0

        return da_T

    if args.part2_tke_analysis == "TRUE" and has_TKE:
        DQKE_T = (2 * ds["DTKE"]).rename("DQKE_T")
        QSHEAR_T =   W2T(ds["QSHEAR"]).rename("QSHEAR_T")
        QBUOY_T  =   W2T(ds["QBUOY"]).rename("QBUOY_T")
        QWT_T    =   2*W2T(ds["QWT"]).rename("QWT_T")
        QDISS_T  = - W2T(ds["QDISS"]).rename("QDISS_T")

        QRES_T = (DQKE_T - (QSHEAR_T + QBUOY_T + QWT_T + QDISS_T)).rename("QRES_T")
        ds = xr.merge([ds, QSHEAR_T, QBUOY_T, QWT_T, QDISS_T, QRES_T, DQKE_T])


    # Virtual temperature
    THETA  = (300 + ds["T"]).rename("THETA")
    THETAV = THETA * (1 + 0.61*ds["QVAPOR"] - ds["QCLOUD"])
    THETAV = THETAV.rename("THETAV")

    ds = xr.merge([ds, THETAV, THETA])
    ds = ds.mean(dim=['time', 'south_north', 'south_north_stag'], keep_attrs=True)

    #ds_ref_stat.mean(dim=['time', 'south_north', 'south_north_stag', "west_east", "west_east_stag"], keep_attrs=True)
    ds_ref_stat = ds.mean(dim=["west_east", "west_east_stag"])


    ref_Z_W = (ds_ref_stat.PHB + ds_ref_stat.PH) / 9.81
    ref_Z_T = (ref_Z_W[1:] + ref_Z_W[:-1]) / 2

    Nx = ds.dims['west_east']
    Nz = ds.dims['bottom_top']

    X_sU = ds.DX * np.arange(Nx+1) / 1e3
    X_sT = (X_sU[1:] + X_sU[:-1]) / 2
    X_T = np.repeat(np.reshape(X_sT, (1, -1)), [Nz,], axis=0)
    X_W = np.repeat(np.reshape(X_sT, (1, -1)), [Nz+1,], axis=0)

    Z_W = (ds.PHB + ds.PH) / 9.81
    Z_T = (Z_W[1:, :] + Z_W[:-1, :]) / 2


    Z_W_idx_1km = np.argmin(np.abs(ref_Z_W.to_numpy() - 1e3)) 
    Z_W_idx_500m = np.argmin(np.abs(ref_Z_W.to_numpy() - 500.0)) 

    Z_T_idx_1km = np.argmin(np.abs(ref_Z_T.to_numpy() - 1e3)) 
    Z_T_idx_500m = np.argmin(np.abs(ref_Z_T.to_numpy() - 500.0)) 


    theta = ds.T + 300.0
    zeta = (ds.V[:, 1:] - ds.V[:, :-1]) / ds.DX


    theta_prime = ds["T"] - ds_ref_stat["T"]
    U_prime = ds["U"] - ds_ref_stat["U"]
    V_prime = ds["V"] - ds_ref_stat["V"]


    U_prime = U_prime.rename("U_prime")
    V_prime = V_prime.rename("V_prime")


    def ddz(da):
       
        da_np = da.to_numpy() 
        dfdz = xr.zeros_like(ds_ref_stat.PH)
       
        z_T = ref_Z_T.to_numpy() 
        dz = z_T[1:] - z_T[:-1]
        dfdz[1:-1] = (da_np[1:] - da_np[:-1]) / dz
        dfdz[0] = np.nan
        dfdz[-1] = np.nan

        return dfdz 
 

    NVfreq2 = (g0 / 300.0 * ddz(ds_ref_stat["THETAV"])).rename("NVfreq2")
    Nfreq2  = (g0 / 300.0 * ddz(ds_ref_stat["THETA"])).rename("Nfreq2")

       
    #NVfreq = ((g0 / 300.0 * ddz(ds_ref_stat["THETAV"]))**0.5).rename("NVfreq")
    #Nfreq = ((g0 / 300.0 * ddz(ds_ref_stat["THETA"]))**0.5).rename("Nfreq")

    #WND = (ds_ref_stat["U"]**2 +  ds_ref_stat["V"]**2)**0.5
    #WND = WND.rename("WND")

    ds_ref_stat = xr.merge([ds_ref_stat, NVfreq2, Nfreq2])

    if args.part1_x_rolling != 1:

        print("Smooth over W with rolling = ", args.part1_x_rolling)

        if args.part1_x_rolling < 1 or args.part1_x_rolling % 2 != 1:
            raise Exception("Bad x_rolling. It should be positive and odd.")

        window = args.part1_x_rolling // 2

        da_W = ds["W"].copy()
        
        for i in range(1, window+1):
            da_W += ds["W"].roll(west_east=i, roll_coords=False) + ds["W"].roll(west_east=-i, roll_coords=False)

        da_W /= 2*window+1
 
        ds["W"][:] = da_W.to_numpy()[:]

    print("Done")

    return(
        dict(
            ds=ds,
            ds_ref_stat=ds_ref_stat,
            ref_Z_W = ref_Z_W,
            ref_Z_T = ref_Z_T,
            X_sU = X_sU,
            X_sT = X_sT,
            X_T = X_T,
            X_W = X_W,
            Z_W = Z_W,
            Z_T = Z_T,
            Z_W_idx_1km = Z_W_idx_1km,
            Z_W_idx_500m = Z_W_idx_500m,
            Z_T_idx_1km = Z_T_idx_1km,
            Z_T_idx_500m = Z_T_idx_500m,

        )
    )

 

data = loadData(args.input_dir)



if base_exists:
    data_base = loadData(args.input_dir_base)
    diff_ds = data["ds"] - data_base["ds"]
else:
    diff_ds = data["ds"]

print("Done loading data.")


# `ref` here means after takin horizontal mean, as a vertical profile
ds = data["ds"]
ds_ref_stat = data["ds_ref_stat"]

if base_exists:
    ds_base = data_base["ds"]
    ds_base_ref_stat = data_base["ds_ref_stat"]
    diff_ds_ref_stat = data["ds_ref_stat"] - data_base["ds_ref_stat"]
else:
    diff_ds_ref_stat = data["ds_ref_stat"]


X_sU = data["X_sU"]
X_sT = data["X_sT"]
X_T = data["X_T"]
X_W = data["X_W"]
Z_W = data["Z_W"]
Z_T = data["Z_T"]
ref_Z_W = data["ref_Z_W"]
ref_Z_T = data["ref_Z_T"]
    
has_TKE = "QKE" in ds

print("Loading matplotlib...")
import matplotlib
if args.no_display:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
print("Done")

# ======= FIRST PART ========

if args.plot_part1 and base_exists:

    ncol = 1
    nrow = 5

    w = [6,]

    figsize, gridspec_kw = tool_fig_config.calFigParams(
        w = w,
        h = [4,] + [4/3,] * (nrow-1),
        wspace = 1.0,
        hspace = 1.0,
        w_left = 1.0,
        w_right = 1.5,
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
    )

    #time_fmt="%y/%m/%d %Hh"

    if args.overwrite_title == "":
        fig.suptitle("%sTime: %d ~ %d hr" % (args.extra_title, relTimeInHrs(time_beg), relTimeInHrs(time_end),))
        
    else:
        fig.suptitle(args.overwrite_title)


    thumbnail_numberings = args.thumbnail_numbering[args.thumbnail_skip_part1:]
    ax_cnt = 0

    def nextAxes():

        global ax_cnt 

        _ax = ax[ax_cnt, 0]
        thumbnail_numbering = thumbnail_numberings[ax_cnt]
        
        ax_cnt += 1

        return _ax, thumbnail_numbering


    u_levs = np.linspace(-1, 1, 11)
    v_levs = np.linspace(-1, 1, 11)
    w_levs = np.linspace(args.part1_W_levs[0], args.part1_W_levs[1], int(args.part1_W_levs[2]))
    tke_levs = np.linspace(args.part1_TKE_levs[0], args.part1_TKE_levs[1], int(args.part1_TKE_levs[2]))
    tke_levs = [-1, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0] #np.linspace(args.TKE_levs[0], args.TKE_levs[1], int(args.TKE_levs[2]))

    QVAPOR_diff_levs = np.arange(-2, 2.5, 0.5)

    cmap_diverge = cmocean.cm.balance 
    cmap_linear = cmocean.cm.matter

    # Version 1: shading = TKE, contour = W

    """
    tke = ds.QKE.to_numpy() / 2
    tke[tke < 0.1] = np.nan
    mappable1 = ax[0, 0].contourf(X_T, Z_T, tke, levels=tke_levs, cmap=cmap_linear, extend="max")
    cax = tool_fig_config.addAxesNextToAxes(fig, ax[0, 0], "right", thickness=0.03, spacing=0.05)
    cbar0 = plt.colorbar(mappable1, cax=cax, orientation="vertical")

    cs = ax[0, 0].contour(X_W, Z_W, ds.W * 1e2, levels=w_levs, colors="black")
    plt.clabel(cs)
    """

    _ax, _thumbnail_numbering = nextAxes()

    # Version 2: shading = W, contour = TKE
    mappable1 = _ax.contourf(X_W, Z_W, diff_ds["W"]*1e2, levels=w_levs, cmap=cmap_diverge, extend="both")
    cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
    cbar0 = plt.colorbar(mappable1, cax=cax, orientation="vertical")

    if has_TKE:
        tke = diff_ds["QKE"].to_numpy() / 2
        cs = _ax.contour(X_T, Z_T, tke, levels=tke_levs, colors="black")
        plt.clabel(cs)

    cs = _ax.contour(X_T, Z_T, diff_ds["QVAPOR"] * 1e3, levels=QVAPOR_diff_levs, colors="yellow")
    plt.clabel(cs)

    _ax.plot(X_sT, ds_base["PBLH"], color="magenta", linestyle=":")
    _ax.plot(X_sT, ds["PBLH"], color="magenta", linestyle="--")


    _ax.set_title("(%s) $\\delta w$ [$\\mathrm{cm} / \\mathrm{s}$]" % (_thumbnail_numbering,))

    # Begin the line plots

    # Thumbnail: Theta_O Theta_A SST
    _ax, _thumbnail_numbering = nextAxes()
    dTA_mean = np.mean(diff_ds["TA"])
    dTO_mean = np.mean(diff_ds["TO"])
    dTSK_mean = np.mean(diff_ds["TSK"])
    _ax.plot(X_sT, diff_ds["TO"] - dTO_mean, color='black', linestyle="-")
    _ax.plot(X_sT, diff_ds["TA"] - dTA_mean, color='black', linestyle="--")
    _ax.plot(X_sT, diff_ds["TSK"] - dTSK_mean, color='red', linestyle=":")

    _ax.set_title("(%s) $\\delta \\mathrm{SST}'$ (red :), $\\delta \\Theta'_O$ (black -), and $\\delta \\Theta'_A$ (black --).\n$ \\left( \\delta \\overline{\\mathrm{SST}}, \\delta \\overline{\\Theta}_O, \\delta \\overline{\\Theta}_A \\right) = \\left( %.2f, %.2f, %.2f \\right) \\, \\mathrm{K}$" % (_thumbnail_numbering, dTSK_mean, dTO_mean, dTA_mean))
    _ax.set_ylabel("[ $ \\mathrm{K}$ ]", color="black")
    _ax.set_ylim(args.part1_SST_rng)



    # Thumbnail: The 500m layer U V
    """
    _ax, _thumbnail_numbering = nextAxes()

    U500 = ds["U_T"].isel(bottom_top=data["Z_T_idx_500m"]) - ds_base["U_T"].isel(bottom_top=data_base["Z_T_idx_500m"])
    V500 = ds["V"].isel(bottom_top=data["Z_T_idx_500m"]) - ds_base["V"].isel(bottom_top=data_base["Z_T_idx_500m"])
    WND500 = ds["WND"].isel(bottom_top=data["Z_T_idx_500m"]) - ds_base["WND"].isel(bottom_top=data_base["Z_T_idx_500m"])

    U500_mean = np.mean(U500)
    V500_mean = np.mean(V500)
    WND500_mean = np.mean(WND500)

    _ax.plot(X_sT, U500 - U500_mean, color="black", linestyle="--")
    _ax.plot(X_sT, V500 - V500_mean, color="black", linestyle=":")
    _ax.plot(X_sT, WND500 - WND500_mean, color="black", linestyle="-")

    _ax.set_title("(%s) $\\delta U_\\mathrm{500m}'$ (-), $\\delta u'_\\mathrm{500m}$ (--), and $\\delta v'_\\mathrm{500m}$ (:).\n $\\delta \\overline{U}_\\mathrm{500m} = %.2f \\, \\mathrm{m} / \\mathrm{s}$. $\\left( \\delta \\overline{u}_{\\mathrm{500m}}, \\delta \\overline{v}_{\\mathrm{500m}}\\right) = \\left( %.2f, %.2f \\right) \\, \\mathrm{m} / \\mathrm{s}$. " % (_thumbnail_numbering, WND500_mean, U500_mean, V500_mean,))
    _ax.set_ylabel("[ $ \\mathrm{m} / \\mathrm{s} $ ]", color="black")
    _ax.set_ylim(args.part1_U500_rng)
    """

    # Thumbnail: The 10m layer U V
    _ax, _thumbnail_numbering = nextAxes()

    U10_mean = np.mean(diff_ds["U10"])
    V10_mean = np.mean(diff_ds["V10"])
    WND10_mean = np.mean(diff_ds["WND10"])
    _ax.plot(X_sT, diff_ds["U10"] - U10_mean, color="black", linestyle="--", label="$\\delta U_{\\mathrm{10m}} - \\delta \\overline{U}_{\\mathrm{10m}}$")
    _ax.plot(X_sT, diff_ds["V10"] - V10_mean, color="black", linestyle=":", label="$\\delta V_{\\mathrm{10m}} - \\delta \\overline{V}_{\\mathrm{10m}}$")
    _ax.plot(X_sT, diff_ds["WND10"] - WND10_mean, color="black", linestyle="-",   label="$\\left| \\vec{U}_{\\mathrm{10m}} \\right| - \\delta \\overline{\\left| \\vec{U}_{\\mathrm{10m}} \\right|}$")

    _ax.set_title("(%s) $\\delta U_\\mathrm{10m}'$ (-), $\\delta u'_\\mathrm{10m}$ (--), and $\\delta v'_\\mathrm{10m}$ (:).\n $\\delta \\overline{U}_\\mathrm{10m} = %.2f \\, \\mathrm{m} / \\mathrm{s}$. $\\left( \\delta \\overline{u}_{\\mathrm{10m}}, \\delta \\overline{v}_{\\mathrm{10m}}\\right) = \\left( %.2f, %.2f \\right) \\, \\mathrm{m} / \\mathrm{s}$. " % (_thumbnail_numbering, WND10_mean, U10_mean, V10_mean,))
    _ax.set_ylabel("[ $ \\mathrm{m} / \\mathrm{s} $ ]", color="black")
    _ax.set_ylim(args.part1_U10_rng)


    """
    # Thumbnail: The 500m layer DIV VOR
    _ax, _thumbnail_numbering = nextAxes()

    DIV500 = ds["DIV"].isel(bottom_top=data["Z_T_idx_500m"]) - ds_base["DIV"].isel(bottom_top=data_base["Z_T_idx_500m"])
    VOR500 = ds["VOR"].isel(bottom_top=data["Z_T_idx_500m"]) - ds_base["VOR"].isel(bottom_top=data_base["Z_T_idx_500m"])

    DIV500_mean = np.mean(DIV500)
    VOR500_mean = np.mean(VOR500)

    PRECIP = diff_ds["PRECIP"]#RAINC"] + diff_ds["RAINNC"]
    PRECIP_mean = np.mean(PRECIP)

    _ax_twinx = _ax.twinx()

    _ax.plot(X_sT, ( DIV500 - DIV500_mean ) * 1e5, color="black", linestyle="-")
    _ax.plot(X_sT, ( VOR500 - VOR500_mean ) * 1e5, color="black", linestyle="--")

    _ax_twinx.plot(X_sT, (PRECIP - PRECIP_mean)*86400, color='black', linestyle=":", label="$P$")

    _ax.set_title("(%s) $\\delta \\mathrm{D}'_\\mathrm{500m}$ (-), $\\delta \\zeta'_\\mathrm{500m}$ (--), and $\\delta P'$ (:).\n$\\left(\\delta \\overline{D}_\\mathrm{500m}, \\delta \\overline{\\zeta}_\\mathrm{500m} \\right) = \\left( %.2f, %.2f \\right) \\times 10^{-5} \\, \\mathrm{s}^{-1}$. $\\delta \\overline{P} = %.2f \\, \\mathrm{mm} / \\mathrm{day} $ " % (_thumbnail_numbering, DIV500_mean*1e5, VOR500_mean*1e5, PRECIP_mean*86400))

    _ax.set_ylabel("[ $ \\times 10^{-5} \\, \\mathrm{s}^{-1}$ ]", color="black")
    _ax_twinx.set_ylabel("$\\delta P'$ [ $ \\mathrm{mm} \\, / \\, \\mathrm{day} $ ]", color="black")
    _ax.set_ylim(args.part1_DIVVOR500_rng)
    _ax_twinx.set_ylim(args.part1_PRECIP_rng)
    """

    # Thumbnail: The 10m layer VOR DIV
    _ax, _thumbnail_numbering = nextAxes()

    VOR10_mean = np.mean(diff_ds["VOR10"])
    DIV10_mean = np.mean(diff_ds["DIV10"])

    PRECIP = diff_ds["PRECIP"]#RAINC"] + diff_ds["RAINNC"]
    PRECIP_mean = np.mean(PRECIP)

    _ax_twinx = _ax.twinx()

    _ax.plot(X_sU[:-1], ( diff_ds["DIV10"] - DIV10_mean ) * 1e5, color="black", linestyle="-")
    _ax.plot(X_sU[:-1], ( diff_ds["VOR10"] - VOR10_mean ) * 1e5, color="black", linestyle="--")
    _ax_twinx.plot(X_sT, (PRECIP - PRECIP_mean)*86400, color='black', linestyle=":", label="$P$")

    _ax.set_title("(%s) $\\delta \\mathrm{D}'_\\mathrm{10m}$ (-), $\\delta \\zeta'_\\mathrm{10m}$ (--), and $\\delta P'$ (:).\n$\\left(\\delta \\overline{D}_\\mathrm{10m}, \\delta \\overline{\\zeta}_\\mathrm{10m} \\right) = \\left( %.2f, %.2f \\right) \\times 10^{-5} \\, \\mathrm{s}^{-1}$. $\\delta \\overline{P} = %.2f \\, \\mathrm{mm} / \\mathrm{day} $ " % (_thumbnail_numbering, DIV10_mean*1e5, VOR10_mean*1e5, PRECIP_mean*86400))

    _ax.set_ylabel("[ $ \\times 10^{-5} \\, \\mathrm{s}^{-1}$ ]", color="black")
    _ax_twinx.set_ylabel("$\\delta P'$ [ $ \\mathrm{mm} \\, / \\, \\mathrm{day} $ ]", color="black")
    _ax.set_ylim(args.part1_DIVVOR10_rng)
    _ax_twinx.set_ylim(args.part1_PRECIP_rng)

    """
    # Thumbnail: The PSFC
    _ax, _thumbnail_numbering = nextAxes()

    PSFC = diff_ds["PSFC"]
    PSFC_mean = np.mean(diff_ds["PSFC"])

    _ax.plot(X_sT, (PSFC - PSFC_mean), color="black", linestyle="-")

    _ax.set_title("(%s) $\\delta \\mathrm{p}'_s (-).$ \n $ \\overline{p_s} = %.2f \\, \\mathrm{Pa}$. " % (_thumbnail_numbering, PSFC_mean,))

    _ax.set_ylabel("[ Pa ]", color="black")
    #_ax.set_ylim(args.part1_DIVVOR10_rng)
    """


    """
    # Precipitation
    _ax, _thumbnail_numbering = nextAxes()
    PRECIP = diff_ds["PRECIP"]#RAINC"] + diff_ds["RAINNC"]
    PRECIP_mean = np.mean(PRECIP)


    W1km = ds["W"].isel(bottom_top_stag=data["Z_W_idx_1km"]) - ds_base["W"].isel(bottom_top_stag=data_base["Z_W_idx_1km"])
    W1km_mean = np.mean(W1km)

    _ax.plot(X_sT, (PRECIP - PRECIP_mean)*86400, color='black', label="$P$")
    _ax_twinx = _ax.twinx()
    _ax_twinx.plot(X_sT, (W1km - W1km_mean) * 1e2, color='black', linestyle="--")

    _ax.set_title("(%s) $\\delta P'$ (-) and $\\delta W'_{\\mathrm{1km}}$ (--). $\\delta \\overline{P} = %.2f \\, \\mathrm{mm} / \\mathrm{day}$, $\\delta \\overline{W}_\\mathrm{1km} = %.1f \\, \\mathrm{cm} / \\mathrm{s}$" % (_thumbnail_numbering, PRECIP_mean*86400, W1km_mean*1e2))

    _ax.set_ylabel("$\\delta P'$ [ $ \\mathrm{mm} \\, / \\, \\mathrm{day} $ ]", color="black")
    _ax_twinx.set_ylabel("$\\delta W'$ [ $ \\mathrm{cm} \\, / \\, \\mathrm{s} $ ]", color="black")
    """

    # vapor
    _ax, _thumbnail_numbering = nextAxes()
    dQO_mean = np.mean(diff_ds["QO"])
    dQA_mean = np.mean(diff_ds["QA"])
    _ax.plot(X_sT, ( diff_ds["QO"] - dQO_mean ) * 1e3, color='black', linestyle="-", label="$\\delta Q_O^* - \\delta \\overline{Q}^*_O$")
    _ax.plot(X_sT, 20*( diff_ds["QA"] - dQA_mean ) * 1e3, color='black', linestyle="--",  label="$\\delta Q_A - \\delta \\overline{Q}_A$")
    _ax.set_title("(%s) $\\delta Q'_O$ (-) and $20 \\times \\delta Q'_A$ (--). $\\left(\\delta \\overline{Q}_O^*, \\delta \\overline{Q}_A \\right) = \\left( %.2f, %.2f \\right)$ g / kg" % (_thumbnail_numbering, dQO_mean*1e3, dQA_mean*1e3))
    _ax.set_ylabel("[ $ \\mathrm{g} \\, / \\, \\mathrm{kg}$ ]", color="black")
    _ax.set_ylim(args.part1_Q_rng)




    for i, _ax in enumerate(ax[:, 0]):
        
        _ax.set_xlabel("$x$ [km]")
        _ax.set_xlim(np.array(args.part1_x_rng))

        if i == 0:
            
            _ax.set_ylim(args.part1_z_rng)
            
            _ax.set_ylabel("$z$ [ km ]")
            yticks = np.array(_ax.get_yticks())
            _ax.set_yticks(yticks, ["%.1f" % _y for _y in yticks/1e3])
            
        _ax.grid(visible=True, which='major', axis='both')
            




    if args.output1 != "":
        print("Saving output: ", args.output1)
        fig.savefig(args.output1, dpi=300)

    if not args.no_display:
        print("Showing figure...")
        plt.show()

# =====================================================================

if args.plot_part2:

    if base_exists:
        delta_word = "\\delta"
    else:
        delta_word = ""



    print("Generating vertical profile")

    ncol = 5
    nrow = 1

    if not has_TKE:
        ncol -= 1

    w = [1.5,] * ncol

    if args.part2_tke_analysis == "TRUE":

        if not has_TKE:
            raise Exception("TKE analysis is required but there is no TKE in the data")

        ncol += 1 
        w = w + [1.5,]


    figsize, gridspec_kw = tool_fig_config.calFigParams(
        w = w,
        h = [4, ],
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
        squeeze=False,
    )


    if args.overwrite_title == "":
        fig.suptitle("%sTime: %d ~ %d hr" % (args.extra_title, relTimeInHrs(time_beg), relTimeInHrs(time_end),))
        
    else:
        fig.suptitle(args.overwrite_title)


    iii = 0
    # Temperature
    _ax = ax[0, iii]
    _ax.plot(diff_ds_ref_stat["THETA"], ref_Z_T, 'k-', label="$\\overline{\\theta}$")
    _ax.plot(diff_ds_ref_stat["THETAV"], ref_Z_T, 'r--', label="$\\overline{\\theta_v}$")
    _ax.set_title("(%s) $ %s \\overline{\\theta}$ (-), $ %s \\overline{\\theta}_v$ (--)" % (args.thumbnail_numbering[args.thumbnail_skip_part2 + iii], delta_word, delta_word,))
    _ax.set_xlabel("[ $\\mathrm{K}$ ]")
    _ax.set_xlim(args.part2_THETA_rng)
    #_ax.legend(loc="upper right")
    iii += 1

    # Stability
    _ax = ax[0, iii]
    _ax.plot(diff_ds_ref_stat["Nfreq2"] * 1e4, ref_Z_W, 'k-', label="$N^2$")
    _ax.plot(diff_ds_ref_stat["NVfreq2"] * 1e4, ref_Z_W, 'r--', label="$N^2_v$")
    _ax.set_title("(%s) $ %s N^2$ (-), $ %s N^2_v$ (--)" % (args.thumbnail_numbering[args.thumbnail_skip_part2 + iii], delta_word, delta_word, ))
    _ax.set_xlabel("[ $\\times 10^{-4} \\mathrm{s}^{-2}$ ]")
    _ax.set_xlim(args.part2_Nfreq2_rng)
    #_ax.legend(loc="upper right")
    iii += 1

    # U, V
    _ax = ax[0, iii]
    _ax.plot(diff_ds_ref_stat["WND"], ref_Z_T, linestyle="-", color="black")
    _ax.plot(diff_ds_ref_stat["U"], ref_Z_T, linestyle="--", color="blue")#, ref_Z_T)
    _ax.plot(diff_ds_ref_stat["V"], ref_Z_T, linestyle=":", color="red")#, ref_Z_T)

    _ax.set_title("(%s) $ %s U $ (-), $ %s \\overline{u}$ (--), $ %s \\overline{v}$ (..)" % (args.thumbnail_numbering[args.thumbnail_skip_part2 + iii], delta_word, delta_word, delta_word, ))
    _ax.set_xlabel("[ $\\mathrm{m} \\, / \\, \\mathrm{s}$ ]")
    _ax.set_xlim(args.part2_U_rng)
    iii += 1


    # Q
    _ax = ax[0, iii]
    _ax.plot(diff_ds_ref_stat["QVAPOR"] * 1e3, ref_Z_T, color="black")

    _ax.set_title("(%s) $ %s \\overline{Q}_\\mathrm{vapor}$" % (args.thumbnail_numbering[args.thumbnail_skip_part2 + iii], delta_word, ))
    _ax.set_xlabel("[ $\\mathrm{g} \\, / \\, \\mathrm{kg}$ ]")
    _ax.set_xlim(args.part2_Q_rng)
    iii += 1

    # TKE

    if has_TKE:
        _ax = ax[0, iii]
        _ax.plot(diff_ds_ref_stat["QKE"]/2, ref_Z_T, color="black")
        _ax.set_title("(%s) $ %s \\overline{\\mathrm{TKE}}$" % (args.thumbnail_numbering[args.thumbnail_skip_part2 + iii], delta_word))
        _ax.set_xlabel("[ $\\mathrm{m}^2 \\, / \\, \\mathrm{s}^2$ ]")
        _ax.set_xlim(args.part2_TKE_rng)
        iii+=1



    if args.part2_tke_analysis == "TRUE":

        # TKE budget
        _ax = ax[0, iii]

        _ax.set_title("(%s) $%s \\mathrm{TKE}$ budget" % (args.thumbnail_numbering[args.thumbnail_skip_part2 + iii], delta_word))
        _ax.plot(diff_ds_ref_stat["DQKE_T"] / 2, ref_Z_T,   color="black", linestyle='-', label="$\\frac{\\partial q}{\\partial t}$")
        _ax.plot(diff_ds_ref_stat["QSHEAR_T"]/2, ref_Z_T, color="red", linestyle='-',   label="$q_{sh}$")
        _ax.plot(diff_ds_ref_stat["QBUOY_T"]/2, ref_Z_T,  color="blue", linestyle='-',  label="$q_{bu}$")
        _ax.plot(diff_ds_ref_stat["QWT_T"]/2 , ref_Z_T,    color="red", linestyle='--',  label="$q_{vt}$")
        _ax.plot(diff_ds_ref_stat["QDISS_T"]/2, ref_Z_T,  color="blue", linestyle='--', label="$q_{ds}$")
        _ax.plot(diff_ds_ref_stat["QRES_T"]/2, ref_Z_T,   color="#aaaaaa", linestyle='--',label="$res$")

        #_ax.plot(ds_ref_stat["QWT_T"]/2 + ds_ref_stat["QADV_T"]/2, ref_Z_T,    color="red", linestyle='--',  label="$q_{vt} + q_{adv}$")

        _ax.set_xlabel("[ $\\mathrm{m}^2 \\, / \\, \\mathrm{s}^2$ ]")
        _ax.set_xlim(args.part2_DTKE_rng)

        _ax.legend(fontsize=12, bbox_to_anchor=(1, 1))#loc="upper right")
        iii+=1

    for _ax in ax[0, :]:
        trans = transforms.blended_transform_factory(_ax.transAxes, _ax.transData)

        if base_exists:
            _ax.plot([0, 1], [ds_base_ref_stat["PBLH"].to_numpy()]*2, color="magenta", linestyle=":", transform=trans)

        _ax.plot([0, 1], [ds_ref_stat["PBLH"].to_numpy()]*2, color="magenta", linestyle="--", transform=trans)



    for i, _ax in enumerate(ax[0, :]):

        _ax.set_ylim(args.part2_z_rng)
         
        _ax.set_ylabel("$z$ [ km ]")
        yticks = np.array(_ax.get_yticks())
        _ax.set_yticks(yticks, ["%.1f" % _y for _y in yticks/1e3])

        _ax.grid(visible=True, which='major', axis='both')


    if args.output2 != "":
        print("Saving output: ", args.output2)
        fig.savefig(args.output2, dpi=300)

    if not args.no_display:
        print("Showing figure...")
        plt.show()


