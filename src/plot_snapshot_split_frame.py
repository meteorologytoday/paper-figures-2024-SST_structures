import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import datetime
import wrf_load_helper 
import cmocean

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-dir', type=str, help='Input directory.', required=True)
parser.add_argument('--input-dir-base', type=str, help='Input directory for base.', default="")
parser.add_argument('--output1', type=str, help='Output filename in png.', default="")
parser.add_argument('--output2', type=str, help='Output filename in png.', default="")
parser.add_argument('--extra-title', type=str, help='Title', default="")
parser.add_argument('--overwrite-title', type=str, help='If set then title will be set to this.', default="")
parser.add_argument('--blh-method', type=str, help='Method to determine boundary layer height', default=[], nargs='+', choices=['bulk', 'grad'])
parser.add_argument('--SST-rng', type=float, nargs=2, help='Title', default=[14.5, 16.5])
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--plot-check', action="store_true")
parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
#parser.add_argument('--ref-time-rng', type=int, nargs=2, help="Reference time range in hours after --exp-beg-time", required=True)
parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
parser.add_argument('--z1-rng', type=float, nargs=2, help='The plotted height rng in meters.', default=[0, 5000.0])
parser.add_argument('--z2-rng', type=float, nargs=2, help='The plotted height rng in meters.', default=[0, 2000.0])
parser.add_argument('--x-rng', type=float, nargs=2, help='The plotted height rng in kilometers', default=[None, None])
parser.add_argument('--U-rng', type=float, nargs=2, help='The plotted height rng in kilometers', default=[None, None])
parser.add_argument('--V-rng', type=float, nargs=2, help='The plotted height rng in kilometers', default=[None, None])
parser.add_argument('--Q-rng', type=float, nargs=2, help='The plotted height rng in kilometers', default=[None, None])
parser.add_argument('--W-levs', type=float, nargs=3, help='The plotted W contours. Three parameters will be fed into numpy.linspace', default=[-8, 8, 17])
parser.add_argument('--TKE-levs', type=float, nargs=3, help='The plotted W contours. Three parameters will be fed into numpy.linspace', default=[0.2, 1, 5])
parser.add_argument('--U10-rng', type=float, nargs=2, help='The plotted surface wind in m/s', default=[None, None])
parser.add_argument('--TKE-rng', type=float, nargs=2, help='The plotted surface wind in m/s', default=[None, None])
parser.add_argument('--DTKE-rng', type=float, nargs=2, help='The plotted surface wind in m/s', default=[None, None])

parser.add_argument('--tke-analysis', type=str, help='analysis beg time', choices=["TRUE", "FALSE"], required=True)
parser.add_argument('--thumbnail-skip-part1', type=int, help='Skip of thumbnail numbering.', default=0)
parser.add_argument('--thumbnail-skip-part2', type=int, help='Skip of thumbnail numbering.', default=0)
parser.add_argument('--thumbnail-numbering', type=str, help='Skip of thumbnail numbering.', default="abcdefghijklmn")
parser.add_argument('--x-rolling', type=int, help='The plotted height rng in kilometers', default=1)

args = parser.parse_args()

print(args)

    
g0 = 9.81
zerodegC = 273.15



if args.input_dir_base != "":
    print("Base exists: ", args.input_dir_base)
    base_exists = True


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

    # Compute TKE stuff
    # Bolton (1980)
    E1 = 0.6112e3 * np.exp(17.67 * (ds["TSK"] - 273.15) / (ds["TSK"] - 29.65) )
    QSFCMR = 0.62175 * E1 / (ds["PSFC"] - E1)
    QA  = ds["QVAPOR"].isel(bottom_top=0).rename("QA")
    QO = QSFCMR.rename("QO")

    ds = xr.merge([ds, QO, QA])


    WND10 = ((ds.U10**2 + ds.V10**2)**0.5).rename("WND10")
    ds = xr.merge([ds, WND10])

    has_TKE = "QKE" in ds


    # TKE variables
    def W2T(da_W):
        
        da_T = xr.zeros_like(ds["T"])
        
        da_W = da_W.to_numpy()
        da_T[:, :, :, :] = ( da_W[:, :-1, :, :] + da_W[:, 1:, :, :] ) / 2.0

        return da_T

    if args.tke_analysis == "TRUE" and has_TKE:
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
        
    NVfreq = ((g0 / 300.0 * ddz(ds_ref_stat["THETAV"]))**0.5).rename("NVfreq")
    Nfreq = ((g0 / 300.0 * ddz(ds_ref_stat["THETA"]))**0.5).rename("Nfreq")

    ds_ref_stat = xr.merge([ds_ref_stat, NVfreq, Nfreq])

    ds = ds.rolling(west_east=args.x_rolling, center=True).mean()

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
        )
    )

 

data = loadData(args.input_dir)

if base_exists:
    data_base = loadData(args.input_dir_base)

print("Done loading data.")

ds = data["ds"]
ds_ref_stat = data["ds_ref_stat"]
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

ncol = 1
nrow = 4

w = [6,]

figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = w,
    h = [4, 4/3, 4/3, 4/3],
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

#time_fmt="%y/%m/%d %Hh"

if args.overwrite_title == "":
    fig.suptitle("%sTime: %.2f ~ %.2f hr" % (args.extra_title, relTimeInHrs(time_beg), relTimeInHrs(time_end),))
    
else:
    fig.suptitle(args.overwrite_title)


u_levs = np.linspace(-1, 1, 11)
v_levs = np.linspace(-1, 1, 11)
w_levs = np.linspace(args.W_levs[0], args.W_levs[1], int(args.W_levs[2]))
tke_levs = np.linspace(args.TKE_levs[0], args.TKE_levs[1], int(args.TKE_levs[2]))

QVAPOR_diff_levs = np.arange(-2, 2, 0.2)

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

ds = ds - data_base["ds"]


# Version 2: shading = W, contour = TKE
mappable1 = ax[0, 0].contourf(X_W, Z_W, ds.W*1e2, levels=w_levs, cmap=cmap_diverge, extend="both")
cax = tool_fig_config.addAxesNextToAxes(fig, ax[0, 0], "right", thickness=0.03, spacing=0.05)
cbar0 = plt.colorbar(mappable1, cax=cax, orientation="vertical")

if has_TKE:
    tke = ds.QKE.to_numpy() / 2
    cs = ax[0, 0].contour(X_T, Z_T, tke, levels=tke_levs, colors="black")
    plt.clabel(cs)

if base_exists:
    #QVAPOR_diff = ds["QVAPOR"] - data_base["ds"]["QVAPOR"]
    QVAPOR_diff = ds["QVAPOR"] 
    cs = ax[0, 0].contour(X_T, Z_T, QVAPOR_diff * 1e3, levels=QVAPOR_diff_levs, colors="yellow")
    plt.clabel(cs)


for _ax in ax[0:1, 0].flatten():
    _ax.plot(X_sT, ds.PBLH, color="magenta", linestyle="--")

for _ax in ax[0, 1:]:
    trans = transforms.blended_transform_factory(_ax.transAxes, _ax.transData)
    _ax.plot([0, 1], [ds_ref_stat["PBLH"].to_numpy()]*2, color="magenta", linestyle="-.", transform=trans)

U10_mean = np.mean(ds.U10)
V10_mean = np.mean(ds.V10)
WND10_mean = np.mean(ds.WND10)
ax[1, 0].plot(X_sT, ds.U10 - U10_mean, color="gray", linestyle="dashed", label="$U_{\\mathrm{10m}} - \\overline{U}_{\\mathrm{10m}}$")
ax[1, 0].plot(X_sT, ds.V10 - V10_mean, color="gray", linestyle="dotted", label="$V_{\\mathrm{10m}} - \\overline{V}_{\\mathrm{10m}}$")
ax[1, 0].plot(X_sT, ds.WND10 - WND10_mean, color="black", linestyle="solid",   label="$\\left| \\vec{U}_{\\mathrm{10m}} \\right| - \\overline{\\left| \\vec{U}_{\\mathrm{10m}} \\right|}$")

# vapor
ax[2, 0].plot(X_sT, ds.QO * 1e3, color='blue', label="$Q_O$")
ax[2, 0].plot(X_sT, ds.QA * 1e3, color='red',  label="$Q_A$")

# SST
ax[3, 0].plot(X_sT, ds["TSK"] - zerodegC, color='blue', label="SST")
ax[3, 0].plot(X_sT, ds.T2 - zerodegC, color='red', label="$T_{\\mathrm{2m}}$")


for i, _ax in enumerate(ax[:, 0]):
    
    _ax.set_xlabel("$x$ [km]")
    _ax.set_xlim(np.array(args.x_rng))

    if i == 0:
        
        _ax.set_ylim(args.z1_rng)
        
        _ax.set_ylabel("$z$ [ km ]")
        yticks = np.array(_ax.get_yticks())
        _ax.set_yticks(yticks, ["%.1f" % _y for _y in yticks/1e3])
        
    _ax.grid(visible=True, which='major', axis='both')
        

thumbnail_numbering = args.thumbnail_numbering[args.thumbnail_skip_part1:]

        
        
ax[0, 0].set_title("(%s) $W$ [$\\mathrm{cm} / \\mathrm{s}$]" % (thumbnail_numbering[0],))
ax[2, 0].set_title("(%s) $Q_O$ (blue) and $Q_A$ (red)" % (thumbnail_numbering[2],))
ax[3, 0].set_title("(%s) SST (blue) and $T_\\mathrm{2m}$ (red)" % (thumbnail_numbering[3],))

ax[1, 0].legend(loc="upper right")
ax[1, 0].set_ylabel("[ $ \\mathrm{m} / \\mathrm{s} $ ]", color="black")
ax[1, 0].set_title("(%s) $\\left( \\overline{U}_{\\mathrm{10m}}, \\overline{V}_{\\mathrm{10m}}\\right) = \\left( %.2f, %.2f \\right)$, $\\overline{\\left|\\vec{U}_{\\mathrm{10m}}\\right|} = %.2f $." % (thumbnail_numbering[1], U10_mean, V10_mean, WND10_mean))

ax[2, 0].set_ylabel("[ $ \\mathrm{g} \\, / \\, \\mathrm{kg}$ ]", color="black")
ax[3, 0].set_ylabel("[ $ \\mathrm{K}$ ]", color="black")


ax[1, 0].set_ylim(args.U10_rng)
ax[2, 0].set_ylim(args.Q_rng)
ax[3, 0].set_ylim(args.SST_rng)

if args.output1 != "":
    print("Saving output: ", args.output1)
    fig.savefig(args.output1, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()

# =====================================================================

print("Generating vertical profile")

ncol = 4
nrow = 1

w = [2, 1, 1, 1]

if args.tke_analysis == "TRUE":

    ncol += 2 
    w = w + [1, 2,]

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
    fig.suptitle("%sTime: %.2f ~ %.2f hr" % (args.extra_title, relTimeInHrs(time_beg), relTimeInHrs(time_end),))
    
else:
    fig.suptitle(args.overwrite_title)


iii = 0
# Temperature
_ax = ax[0, iii]
_ax.plot(ds_ref_stat["T"] + 300, ref_Z_T, 'k-', label="$\\overline{\\theta}$")
_ax.plot(ds_ref_stat["THETAV"], ref_Z_T, 'r--', label="$\\overline{\\theta_v}$")
_ax.set_title("(%s) $\\overline{\\theta}$ and $\\overline{\\theta}_v$" % (args.thumbnail_numbering[args.thumbnail_skip_part2 + iii],))
_ax.set_xlabel("[ $\\mathrm{K}$ ]")
_ax.set_xlim([285, 300])
_ax.legend(loc="upper right")
iii += 1

# Stability
_ax = ax[0, iii]
_ax.plot(ds_ref_stat["Nfreq"], ref_Z_W, 'k-', label="$N$")
_ax.plot(ds_ref_stat["NVfreq"], ref_Z_W, 'r--', label="$N_v$")
_ax.set_title("(%s) $N$ and $N_v$" % (args.thumbnail_numbering[args.thumbnail_skip_part2 + iii],))
_ax.set_xlabel("[ $\\mathrm{s}^{-1}$ ]")
_ax.set_xlim([0, 0.03])
_ax.legend(loc="upper right")
iii += 1

# U
_ax = ax[0, iii]
_ax.plot(ds_ref_stat["U"], ref_Z_T)

_ax.set_title("(%s) $U$" % (args.thumbnail_numbering[args.thumbnail_skip_part2 + iii],))
_ax.set_xlabel("[ $\\mathrm{m} \\, / \\, \\mathrm{s}$ ]")
_ax.set_xlim(args.U_rng)
iii += 1


# V
_ax = ax[0, iii]
_ax.plot(ds_ref_stat["V"], ref_Z_T)

_ax.set_title("(%s) $V$" % (args.thumbnail_numbering[args.thumbnail_skip_part2 + iii],))
_ax.set_xlabel("[ $\\mathrm{m} \\, / \\, \\mathrm{s}$ ]")
_ax.set_xlim(args.V_rng)
iii += 1


if args.tke_analysis == "TRUE":
    # TKE
    _ax = ax[0, iii]
    _ax.plot(ds_ref_stat["QKE"]/2, ref_Z_T)
    _ax.set_title("(%s) TKE" % (args.thumbnail_numbering[args.thumbnail_skip_part2 + iii],))
    _ax.set_xlabel("[ $\\mathrm{m}^2 \\, / \\, \\mathrm{s}^2$ ]")
    _ax.set_xlim(args.TKE_rng)
    iii+=1

    # TKE budget
    _ax = ax[0, iii]

    _ax.set_title("(%s) TKE budget" % (args.thumbnail_numbering[args.thumbnail_skip_part2 + iii],))
    _ax.plot(ds_ref_stat["DQKE_T"] * 1e1 / 2, ref_Z_T,   color="black", linestyle='solid', label="$10 \\times \\partial q / \\partial t$")
    _ax.plot(ds_ref_stat["QSHEAR_T"]/2, ref_Z_T, color="red", linestyle='solid',   label="$q_{sh}$")
    _ax.plot(ds_ref_stat["QBUOY_T"]/2, ref_Z_T,  color="blue", linestyle='solid',  label="$q_{bu}$")
    _ax.plot(ds_ref_stat["QWT_T"]/2 , ref_Z_T,    color="red", linestyle='dashed',  label="$q_{vt}$")
    _ax.plot(ds_ref_stat["QDISS_T"]/2, ref_Z_T,  color="blue", linestyle='dashed', label="$q_{ds}$")
    _ax.plot(ds_ref_stat["QRES_T"]/2, ref_Z_T,   color="#aaaaaa", linestyle='dashed',label="$res$")

    #_ax.plot(ds_ref_stat["QWT_T"]/2 + ds_ref_stat["QADV_T"]/2, ref_Z_T,    color="red", linestyle='dashed',  label="$q_{vt} + q_{adv}$")

    _ax.set_xlabel("[ $\\mathrm{m}^2 \\, / \\, \\mathrm{s}^2$ ]")
    _ax.set_xlim(args.DTKE_rng)

    _ax.legend(loc="upper right")
    iii+=1

for _ax in ax[0, :]:
    trans = transforms.blended_transform_factory(_ax.transAxes, _ax.transData)
    _ax.plot([0, 1], [ds_ref_stat["PBLH"].to_numpy()]*2, color="magenta", linestyle="--", transform=trans)


for i, _ax in enumerate(ax[0, :]):

    _ax.set_ylim(args.z2_rng)
     
    _ax.set_ylabel("$z$ [ km ]")
    yticks = np.array(_ax.get_yticks())
    _ax.set_yticks(yticks, ["%d" % _y for _y in yticks/1e3])

    _ax.grid(visible=True, which='major', axis='both')


if args.output2 != "":
    print("Saving output: ", args.output2)
    fig.savefig(args.output2, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()


