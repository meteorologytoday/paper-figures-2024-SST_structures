import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import datetime
import wrf_load_helper 
import cmocean
import colorblind


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-dirs', type=str, nargs="+", help='Input directory.', required=True)
parser.add_argument('--input-dirs-base', nargs="*", type=str, help='Input directories for baselines. They can be empty.', default=None)
parser.add_argument('--output', type=str, help='Output filename in png.', default="")

parser.add_argument('--linestyles', nargs="+", type=str, help='Line styles.', required=True)
parser.add_argument('--linecolors', nargs="+", type=str, help='Line styles.', required=True)
parser.add_argument('--labels', nargs="*", type=str, help='Exp names.', default=None)

parser.add_argument('--extra-title', type=str, help='Title', default="")
parser.add_argument('--overwrite-title', type=str, help='If set then title will be set to this.', default="")
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
#parser.add_argument('--ref-time-rng', type=int, nargs=2, help="Reference time range in hours after --exp-beg-time", required=True)
parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
parser.add_argument('--z-rng', type=float, nargs=2, help='The plotted height rng in meters.', default=[0, 2000.0])
parser.add_argument('--U-rng', type=float, nargs=2, help='The plotted height rng in kilometers', default=[None, None])
parser.add_argument('--V-rng', type=float, nargs=2, help='The plotted height rng in kilometers', default=[None, None])
parser.add_argument('--Q-rng', type=float, nargs=2, help='The plotted height rng in kilometers', default=[None, None])
parser.add_argument('--TKE-rng', type=float, nargs=2, help='The plotted surface wind in m/s', default=[None, None])
parser.add_argument('--thumbnail-skip', type=int, help='Skip of thumbnail numbering.', default=0)
parser.add_argument('--thumbnail-numbering', type=str, help='Skip of thumbnail numbering.', default="abcdefghijklmn")
parser.add_argument('--varnames', type=str, nargs="+", help='Variable names.', required=True)
parser.add_argument('--avg-interval', type=float, help='Avg interval in minutes.', required=True)

args = parser.parse_args()

print(args)

g0 = 9.81

exp_beg_time = pd.Timestamp(args.exp_beg_time)
wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)

time_beg = exp_beg_time + pd.Timedelta(minutes=args.time_rng[0])
time_end = exp_beg_time + pd.Timedelta(minutes=args.time_rng[1])

avg_interval = pd.Timedelta(minutes=args.avg_interval)
print("avg_interval = ", str(avg_interval))
base_exists = args.input_dirs_base is not None
if base_exists:
    if len(args.input_dirs_base) != len(args.input_dirs):
        raise Exception("Error: If `--input-dirs-base` is non-empty, it should have the same number of elements as `--input-dirs`.")


def horDecomp(da, name_m="mean", name_p="prime"):
    m = da.mean(dim="west_east").rename(name_m)
    p = (da - m).rename(name_p) 
    return m, p


def relTimeInHrs(t):
    return (t - exp_beg_time.to_datetime64()) / np.timedelta64(1, 'h')

def W2T(ds, da_W):
    
    da_T = xr.zeros_like(ds["T"])
    
    da_W = da_W.to_numpy()
    da_T[:, :, :, :] = ( da_W[:, :-1, :, :] + da_W[:, 1:, :, :] ) / 2.0

    return da_T



def ddz(da, z_W):
   
    da_np = da.to_numpy() 
    dfdz = xr.zeros_like(z_W)
   
    z_T = (z_W[1:] + z_W[:-1]) / 2
    z_T = z_T.to_numpy() 
    dz = z_T[1:] - z_T[:-1]

    dfdz[1:-1] = (da_np[1:] - da_np[:-1]) / dz
    dfdz[0] = np.nan
    dfdz[-1] = np.nan

    return dfdz 
 
def loadData(input_dir):
 
    wsm = wrf_load_helper.WRFSimMetadata(
        start_datetime  = exp_beg_time,
        data_interval   = wrfout_data_interval,
        frames_per_file = args.frames_per_wrfout_file,
    )



    ds = wrf_load_helper.loadWRFDataFromDir(
        wsm, 
        input_dir,
        beg_time = time_beg,
        end_time = time_end,
        prefix="wrfout_d01_",
        avg = avg_interval,
        verbose=False,
        inclusive="left",
    )

    # Virtual temperature
    THETA  = (300 + ds["T"]).rename("THETA")
    THETAV = THETA * (1 + 0.61*ds["QVAPOR"] - ds["QCLOUD"])
    THETAV = THETAV.rename("THETAV")

    W_T = W2T(ds, ds["W"]).rename("W_T")


    MU_FULL = ds.MU + ds.MUB
    RHOD = - MU_FULL * ds.DNW / g0  # notice that DNW is negative, so we need to multiply by -1
    RHOD = RHOD.rename("RHOD").transpose("time", "bottom_top", "south_north", "west_east")

    RHODQVAPOR = RHOD * ds["QVAPOR"]
    
    W_mean, W_prime = horDecomp(W_T, name_m="mean", name_p="prime")
    RHODQVAPOR_mean, RHODQVAPOR_prime = horDecomp(RHODQVAPOR, name_m="mean", name_p="prime")

    WpRHODQVAPORp_mean = ( W_prime * RHODQVAPOR_prime ).mean(dim=["west_east"]).rename("WpRHODQVAPORp_mean")
    

    Z_W = (ds.PHB + ds.PH) / g0
    #dZ_T = ( Z_W.shift(bottom_top_stag=-1) - Z_W ).sel(bottom_top_stag=slice(0, -1)).rename_dims({'bottom_top_stag':'bottom_top'})
    dZ_W = ( Z_W.shift(bottom_top_stag=-1) - Z_W.shift(bottom_top_stag=1) ) / 2
    dRHODQVAPORdz_W = xr.zeros_like(dZ_W)

    RHODQVAPOR_np = RHODQVAPOR.to_numpy()
    dRHODQVAPORdz_W[:, 1:-1, :, :] = (RHODQVAPOR_np[:, 1:, :, :] - RHODQVAPOR_np[:, :-1, :, :] ) / dZ_W.to_numpy()[:, 1:-1, :, :]
    dRHODQVAPORdz_W = dRHODQVAPORdz_W.rename("dRHODQVAPORdz_W")
    
    RHODQVAPOR_vt =  (W2T( ds, - dRHODQVAPORdz_W * ds["EXCH_H"] )).rename("RHODQVAPOR_vt")
   
     
    WpRHODQVAPORp_mean = ( W_prime * RHODQVAPOR_prime ).mean(dim=["west_east"]).rename("WpRHODQVAPORp_mean")

    RHODQVAPOR_vt_ttl = RHODQVAPOR_vt + WpRHODQVAPORp_mean
    RHODQVAPOR_vt_ttl = RHODQVAPOR_vt_ttl.rename("RHODQVAPOR_vt_ttl")

    ps = 1e5
    PRES = ds.P + ds.PB
    T_TOTAL = THETA * (PRES/ps)**(2/7) 
    T_TOTAL = T_TOTAL.rename("T_TOTAL")
    e_SAT = 0.6112e3 * np.exp( 17.67 * (T_TOTAL - 273.15) / (T_TOTAL - 29.65) )

    # P V = n R T
    # P = (w / V/ M) R T
    # P = w/V * R/M * T
    # P = rho * R_w * T

    R_uni = 8.3145
    M_w = 0.018 
    R_w = R_uni / M_w 
    e_w = (ds["RHO"] * ds["QVAPOR"]) * R_w * T_TOTAL
    
    RH = e_w / e_SAT # ???
    RH = RH.rename("RH")

    ds = xr.merge([ds, THETAV, THETA, WpRHODQVAPORp_mean, RHODQVAPOR_vt, RHODQVAPOR_vt_ttl, RH, T_TOTAL])
    ds = ds.mean(dim=['time', 'south_north', 'south_north_stag', 'west_east', 'west_east_stag'], keep_attrs=True)




    Z_W = (ds.PHB + ds.PH) / g0
    Z_T = (Z_W[1:] + Z_W[:-1]) / 2

    NVfreq = ((g0 / 300.0 * ddz(ds["THETAV"], Z_W))**0.5).rename("NVfreq")
    Nfreq = ((g0 / 300.0 * ddz(ds["THETA"], Z_W))**0.5).rename("Nfreq")

    ds = xr.merge([
        ds,
        NVfreq,
        Nfreq,
    ])

  
    ds = xr.merge([ ds[varname] for varname in args.varnames ]) 


    return dict(
        ds = ds,
        Z_W = Z_W,
        Z_T = Z_T,
    )

   
    
data = []
data_base = []

for i, input_dir in enumerate(args.input_dirs):
    
    # Loading data
    print("[%d] Loading wrf dir: %s" % (i, input_dir,))
    
    
    data.append(loadData(input_dir))
    
    
    if base_exists:
        
        input_dir_base = args.input_dirs_base[i]
    
        print("[%d - BASE] Loading wrf dir: %s" % (i, input_dir_base,))
        data_base.append(loadData(input_dir_base))
    
    

# Compute TKE stuff

"""
# Bolton (1980)
E1 = 0.6112e3 * np.exp(17.67 * (ds["TSK"] - 273.15) / (ds["TSK"] - 29.65) )
QSFCMR = 0.62175 * E1 / (ds["PSFC"] - E1)
QA  = ds["QVAPOR"].isel(bottom_top=0).rename("QA")
QO = QSFCMR.rename("QO")
"""


print("Done")


plot_varnames = args.varnames #["QVAPOR", "T", "EXCH_H"]
Nvars = len(plot_varnames)
   
plot_infos = dict(

    W = dict(
        factor = 1e2,
        label = "Vertical velocity $W$",
        unit = "$ \\mathrm{cm} / \\mathrm{s} $",
        lim = None,
        lim_diff = None,
    ),


    QVAPOR = dict(
        factor = 1e3,
        label = "$\\overline{Q}$",
        unit = "$ \\mathrm{g} / \\mathrm{kg} $",
        lim = None,
        lim_diff = None,
    ),

    QCLOUD = dict(
        factor = 1e3,
        label = "$\\overline{Q}_\\mathrm{CLD}$",
        unit = "$ \\mathrm{g} / \\mathrm{kg} $",
        lim = None,
        lim_diff = None,
    ),

    QRAIN = dict(
        factor = 1e3,
        label = "QRAIN",
        unit = "$ \\mathrm{g} / \\mathrm{kg} $",
        lim = None,
        lim_diff = None,
    ),



    EXCH_H = dict(
        factor = 1,
        label = "Scalar vertical diffusivity",
        unit = "$ \\mathrm{m}^2 / \\mathrm{s} $",
        lim = None,
        lim_diff = None,
    ),

    H_DIABATIC = dict(
        factor = 1,
        label = "Diabatic Heating",
        unit = "$ \\mathrm{K} / \\mathrm{s} $",
        lim = None,
        lim_diff = None,
    ),


    THETA = dict(
        factor = 1,
        label = "$\\overline{\\theta}$",
        unit = "$ \\mathrm{K} $",
        lim = [285, 305],
        lim_diff = None,
    ),


    THETAV = dict(
        factor = 1,
        label = "$\\overline{\\theta}_v$",
        unit = "$ \\mathrm{K} $",
        lim = [285, 305],
        lim_diff = None,
    ),

    T_TOTAL = dict(
        factor = 1,
        offset = 273.15,
        label = "$\\overline{T}$",
        unit = "$ {}^\\circ \\mathrm{C} $",
        lim = [-25, 20],
        lim_diff = None,
    ),


    WpRHODQVAPORp_mean = dict(
        factor = 1.0,
        label = "$\\overline{W' \\rho_{w}'}$",
        unit = "$  \\mathrm{kg} / \\mathrm{s} / \\mathrm{m}^2 $",
        lim = None,
        lim_diff = None,
    ),



    RHODQVAPOR_vt = dict(
        factor = 1.0,
        label = "$ - K_Q \\left( \\frac{\\partial \\overline{\\rho_w} }{\\partial z} \\right)$",
        unit = "$  \\mathrm{kg} / \\mathrm{s} / \\mathrm{m}^2 $",
        lim = None,
        lim_diff = None,
    ),

    RHODQVAPOR_vt_ttl = dict(
        factor = 1.0,
        label = "$ \\overline{\\rho_{w}' w'} - K \\left( \\frac{\\partial \\overline{\\rho_w} }{\\partial z} \\right)$",
        unit = "$  \\mathrm{kg} / \\mathrm{s} / \\mathrm{m}^2 $",
        lim = None,
        lim_diff = None,
    ),


    RH = dict(
        factor = 100.0,
        label = "$\\overline{\\mathrm{RH}}$",
        unit = "$ \\% $",
        lim = [0, 100],
        lim_diff = None,
    ),


)

print("Loading matplotlib...")
import matplotlib
if args.no_display:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
print("Done")

print("Generating vertical profile")



ncol = Nvars
nrow = 1

w = [4, ] * ncol

figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = w,
    h = [4, ] * nrow,
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
    fig.suptitle("%sTime: %.2f ~ %.2f hr" % (args.extra_title, relTimeInHrs(time_beg), relTimeInHrs(time_end),))
    
else:
    fig.suptitle(args.overwrite_title)



for i, varname in enumerate(plot_varnames):
    
    _ax = ax[0, i]
    
    plot_info = plot_infos[varname]

    factor = plot_info["factor"] if "factor" in plot_info else 1.0
    offset = plot_info["offset"] if "offset" in plot_info else 0.0
    unit   = plot_info["unit"]


    if base_exists:
        lim   = plot_info["lim_diff"] if "lim_diff" in plot_info else None
        label = plot_info["label_diff"] if "label_diff" in plot_info else "$\\delta$%s" % (plot_info["label"],)

    else:
        lim   = plot_info["lim"] if "lim" in plot_info else None
        label = plot_info["label"]


    
    for j, _data in enumerate(data):


        ds = _data["ds"]
        Z_W = _data["Z_W"]
        Z_T = _data["Z_T"]


        linecolor = args.linecolors[j]
        linestyle = args.linestyles[j]

        #if args.labels is not None:
        #    label = args.labels[i]
    
        plot_data = (ds[varname] - offset) * factor

        if base_exists:
            _data_base = data_base[j]
            
            plot_data = plot_data - (_data_base["ds"][varname] - offset) * factor

        if len(plot_data) == len(Z_T):
            plot_Z = Z_T
        elif len(plot_data) == len(Z_W):
            plot_Z = Z_W


        _ax.plot(plot_data, plot_Z, color=linecolor, linestyle=linestyle, label=label)


    if lim is not None:
        _ax.set_xlim(lim)

    _ax.set_title("(%s) %s" % (args.thumbnail_numbering[args.thumbnail_skip + i], label))
    _ax.set_xlabel("[ %s ]" % unit)
    
    _ax.set_ylim(args.z_rng)
    _ax.set_ylabel("$z$ [ km ]")
    
    yticks = np.array(_ax.get_yticks())
    #_ax.set_yticks(yticks, ["%d" % _y for _y in yticks/1e3])

    _ax.grid(visible=True, which='major', axis='both')


if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()


