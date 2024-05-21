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
parser.add_argument('--output', type=str, help='Output filename in png.', default="")
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
parser.add_argument('--z-rng', type=float, nargs=2, help='The plotted height rng in meters.', default=[0, 2000.0])
parser.add_argument('--x-rng', type=float, nargs=2, help='The plotted height rng in kilometers', default=[None, None])
parser.add_argument('--U-rng', type=float, nargs=2, help='The plotted height rng in kilometers', default=[None, None])
parser.add_argument('--V-rng', type=float, nargs=2, help='The plotted height rng in kilometers', default=[None, None])
parser.add_argument('--U10-rng', type=float, nargs=2, help='The plotted surface wind in m/s', default=[None, None])
parser.add_argument('--TKE-rng', type=float, nargs=2, help='The plotted surface wind in m/s', default=[None, None])
parser.add_argument('--DTKE-rng', type=float, nargs=2, help='The plotted surface wind in m/s', default=[None, None])

parser.add_argument('--tke-analysis', type=str, help='analysis beg time', choices=["TRUE", "FALSE"], required=True)

args = parser.parse_args()

print(args)

Ri_c = dict(
    bulk = 0.25,
    grad = 0.25,
)

exp_beg_time = pd.Timestamp(args.exp_beg_time)
wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)

time_beg = exp_beg_time + pd.Timedelta(minutes=args.time_rng[0])
time_end = exp_beg_time + pd.Timedelta(minutes=args.time_rng[1])

#ref_time_beg = exp_beg_time + pd.Timedelta(minutes=args.ref_time_rng[0])
#ref_time_end = exp_beg_time + pd.Timedelta(minutes=args.ref_time_rng[1])


def relTimeInHrs(t):
    return (t - exp_beg_time.to_datetime64()) / np.timedelta64(1, 'h')



def horDecomp(da, name_m="mean", name_p="prime"):
    m = da.mean(dim="west_east").rename(name_m)
    p = (da - m).rename(name_p) 
    return m, p


# Loading data
print("Loading wrf dir: %s" % (args.input_dir,))

wsm = wrf_load_helper.WRFSimMetadata(
    start_datetime  = exp_beg_time,
    data_interval   = wrfout_data_interval,
    frames_per_file = args.frames_per_wrfout_file,
)



ds = wrf_load_helper.loadWRFDataFromDir(
    wsm, 
    args.input_dir,
    beg_time = time_beg,
    end_time = time_beg,#time_end,
    prefix="wrfout_d01_",
    avg = "ALL",
#    avg = None,
    verbose=False,
    inclusive="both",
)

g0 = 9.81

# Compute TKE stuff
"""
U_U = ds["U"].to_numpy()
W_W = ds["W"].to_numpy()
U_T = ds["T"].copy()
W_T = ds["T"].copy()

U_T[:, :, :, :] = (U_U[:, :, :, 1:] + U_U[:, :, :, :-1]) / 2.0
W_T[:, :, :, :] = (W_W[:, 1:, :, :]  + W_W[:, :-1, :, :]) / 2.0
U_T_mean, U_T_prime = horDecomp( U_T, name_m="U_T_mean", name_p="U_T_prime")
V_T_mean, V_T_prime = horDecomp( ds["V"], name_m="V_T_mean", name_p="V_T_prime")
_, W_T_prime = horDecomp( W_T, name_m="W_T_mean", name_p="W_T_prime")
_, RHO_T_prime = horDecomp( ds["RHO"], name_m="RHO_T_mean", name_p="RHO_T_prime")


U_W_prime = (U_T_prime * W_T_prime).rename("U_W_prime")
V_W_prime = (V_T_prime * W_T_prime).rename("V_W_prime")
RHO_T_prime = (RHO_T_prime * W_T_prime).rename("RHO_T_prime")
TKE = ((U_T_prime**2 + V_T_prime**2 + W_T_prime**2)/2.0).rename("TKE")

ds = xr.merge([ds, U_W_prime, V_W_prime, RHO_T_prime, TKE])
ds = ds.mean(dim=['time', 'south_north', 'south_north_stag'], keep_attrs=True)
"""


# TKE variables
def W2T(da_W):
    
    da_T = xr.zeros_like(ds["T"])
    
    da_W = da_W.to_numpy()
    da_T[:, :, :, :] = ( da_W[:, :-1, :, :] + da_W[:, 1:, :, :] ) / 2.0

    return da_T

if args.tke_analysis == "TRUE":
    DQKE_T = (2 * ds["DTKE"]).rename("DQKE_T")
    QSHEAR_T =   W2T(ds["QSHEAR"]).rename("QSHEAR_T")
    QBUOY_T  =   W2T(ds["QBUOY"]).rename("QBUOY_T")
    QWT_T    =   2*W2T(ds["QWT"]).rename("QWT_T")
    QDISS_T  = - W2T(ds["QDISS"]).rename("QDISS_T")

    QRES_T = (DQKE_T - (QSHEAR_T + QBUOY_T + QWT_T + QDISS_T)).rename("QRES_T")
    ds = xr.merge([ds, QSHEAR_T, QBUOY_T, QWT_T, QDISS_T, QRES_T, DQKE_T])


# Virtual temperature
THETA  = (300 + ds["T"]).rename("THETA")
THETAV = THETA * (1 + 0.61*ds["QVAPOR"])
THETAV = THETAV.rename("THETAV")

ds = xr.merge([ds, THETAV, THETA])
ds = ds.mean(dim=['time', 'south_north', 'south_north_stag'], keep_attrs=True)

#ds_ref_stat.mean(dim=['time', 'south_north', 'south_north_stag', "west_east", "west_east_stag"], keep_attrs=True)
ds_ref_stat = ds.mean(dim=["west_east", "west_east_stag"])

print("Done")

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

P_total = ds.P + ds.PB

P_sfc = P_total.isel(bottom_top=3)
dP_sfcdx = - (P_sfc[1:] - P_sfc[:-1]) / ds.DX

zerodegC = 273.15
theta = ds.T + 300.0
zeta = (ds.V[:, 1:] - ds.V[:, :-1]) / ds.DX
SST = ds.TSK - zerodegC

delta = ds.TH2 - ds.TSK


theta_prime = ds["T"] - ds_ref_stat["T"]
U_prime = ds["U"] - ds_ref_stat["U"]
V_prime = ds["V"] - ds_ref_stat["V"]

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

print(Nfreq)
print(ref_Z_W)
print(ref_Z_T)

ds_ref_stat = xr.merge([ds_ref_stat, NVfreq, Nfreq])


"""
print("Compute boundary layer height")
bl = dict()

for method in args.blh_method:
   
    _Ri_c = Ri_c[method] 
    _blh = []
    _Ri = np.zeros((Nz+1, Nx,))
    for i in range(Nx):

        U = (ds.U[:, i+1] + ds.U[:, i]) / 2

        r = diagnostics.getBoundaryLayerHeight(
            U.to_numpy(),
            ds.V[:, i].to_numpy(),
            theta[:, i].to_numpy(),
            ds.QVAPOR[:, i].to_numpy(),
            Z_W[:, i].to_numpy(),
            Ri_c = _Ri_c,
            method=method,
            skip=1,
            debug=True,
        )

        for j, __blh in enumerate(r[0]):
            _blh.append([X_sT[i], __blh])

        _Ri[:, i] = r[1]['Ri']

    _blh = np.array(_blh)

    bl[method] = dict(
        blh = _blh,
        Ri  = _Ri,
    )
"""

print("Loading matplotlib...")
import matplotlib
if args.no_display:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
print("Done")

if args.plot_check:

    loc = 250    
    U = (ds.U[:, loc+1] + ds.U[:, loc]) / 2
    h, debug_info = diagnostics.getBoundaryLayerHeight(
        U.to_numpy(),
        ds.V[:, loc].to_numpy(),
        theta[:, loc].to_numpy(),
        ds.QVAPOR[:, loc].to_numpy(),
        Z_W[:, loc],
        Ri_c = 0.25,
        method = args.blh_method,
        skip = 0,
        debug = True,
    )


    fig, ax = plt.subplots(1, 7, figsize=(6, 14), sharey=True)


    ax[0].plot(U,  Z_T[:, loc], label="U")
    ax[0].plot(ds.V[:, loc],  Z_T[:, loc], label="V")
    ax[1].plot(theta[:, loc], Z_T[:, loc], label="$\\theta$")


    if args.blh_method == "bulk":
        ax[2].plot(debug_info['theta_v'], Z_W[:, loc], label="$\\theta_v$")
        ax[3].plot(debug_info['dtheta_v'], Z_W[:, loc], label="$\\partial \\theta_v / \\partial z$")
        ax[4].plot(debug_info['du'], Z_W[:, loc], label="$u_z$")
        ax[4].plot(debug_info['dv'], Z_W[:, loc], label="$v_z$")



    elif args.blh_method == "grad":
        ax[2].plot(debug_info['theta_v'], Z_W[:, loc], label="$\\theta_v$")
        ax[3].plot(debug_info['dtheta_vdz'], Z_W[:, loc], label="$\\partial \\theta_v / \\partial z$")
        ax[4].plot(debug_info['dudz'], Z_W[:, loc], label="$u_z$")
        ax[4].plot(debug_info['dvdz'], Z_W[:, loc], label="$v_z$")
        
    else:
        raise Exception("Unknown keyword: %s" % (args.blh_method,))

    ax[5].plot(debug_info['Ri'], Z_W[:, loc], label="$R_i$", linestyle="dotted")
    ax[5].plot(Ri[:, loc], Z_W[:, loc], label="$R_i$ - 2", linestyle="dashed")
    ax[5].set_xlim([-1, 1])

    for _ax in ax.flatten():
        _ax.legend(loc="upper right")
        _ax.grid()

    ax[0].set_ylim(args.z_rng)

    plt.show(block=False)

ncol = 5
nrow = 3

w = [6,] + [2, ] + [1,] * (ncol-2)

if args.tke_analysis == "TRUE":

    ncol += 2 
    w = w + [1, 2,]

figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = w,
    h = [4, 4/3, 4/3],
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
w_levs = np.linspace(-8, 8, 17) / 10
#theta_levs = np.arange(273, 500, 2)
theta_levs = np.concatenate( 
    (
        np.arange(-10, 0, 0.2),
        np.arange(0.2, 10, 0.2),
    )
)

spacing = 0.2
theta_levs = np.arange(-10, 10, spacing) + spacing / 2.0

cmap_diverge = cmocean.cm.balance

mappable1 = ax[0, 0].contourf(X_W, Z_W, ds.W * 1e2, levels=w_levs, cmap=cmap_diverge, extend="both")

#cax = tool_fig_config.addAxesNextToAxes(fig, ax[0, 0], "right", thickness=0.03, spacing=0.05)
#cbar0 = plt.colorbar(mappable1, cax=cax, orientation="vertical")

cs = ax[0, 0].contour(X_W, Z_W, ds.W * 1e2, levels=w_levs, colors="black")
plt.clabel(cs)

for _ax in ax[0:1, 0].flatten():
    #cs = _ax.contour(X_T, Z_T, theta_prime, levels=theta_levs, colors='k')
    #plt.clabel(cs)
    _ax.plot(X_sT, ds.PBLH, color="pink", linestyle="-.")

iii = 1

# Temperature
_ax = ax[0, iii]; iii+=1
_ax.plot(ds_ref_stat["T"] + 300, ref_Z_T, 'k-', label="$\\overline{\\theta}$")
_ax.plot(ds_ref_stat["THETAV"], ref_Z_T, 'r--', label="$\\overline{\\theta_v}$")
_ax.set_title("(d) $\\overline{\\theta}$ and $\\overline{\\theta_v}$")
_ax.set_xlabel("[ $\\mathrm{K}$ ]")
_ax.set_xlim([285, 300])
_ax.legend(loc="upper right")


# Stability
_ax = ax[0, iii]; iii+=1
_ax.plot(ds_ref_stat["Nfreq"], ref_Z_W, 'k-', label="$N$")
_ax.plot(ds_ref_stat["NVfreq"], ref_Z_W, 'r--', label="$N_v$")
_ax.set_title("(e) $N$ and $N_v$")
_ax.set_xlabel("[ $\\mathrm{s}^{-1}$ ]")
_ax.set_xlim([0, 0.02])
_ax.legend(loc="upper right")

# U
_ax = ax[0, iii]; iii+=1
_ax.plot(ds_ref_stat["U"], ref_Z_T)

_ax.set_title("(f) $U$")
_ax.set_xlabel("[ $\\mathrm{m} \\, / \\, \\mathrm{s}$ ]")
_ax.set_xlim(args.U_rng)


# V
_ax = ax[0, iii]; iii+=1
_ax.plot(ds_ref_stat["V"], ref_Z_T)

_ax.set_title("(g) $V$")
_ax.set_xlabel("[ $\\mathrm{m} \\, / \\, \\mathrm{s}$ ]")
_ax.set_xlim(args.V_rng)


if args.tke_analysis == "TRUE":
    # TKE
    _ax = ax[0, iii]; iii+=1
    _ax.plot(ds_ref_stat["QKE"]/2, ref_Z_T)
    _ax.set_title("(h) TKE")
    _ax.set_xlabel("[ $\\mathrm{m}^2 \\, / \\, \\mathrm{s}^2$ ]")
    _ax.set_xlim(args.TKE_rng)

    # TKE budget
    _ax = ax[0, iii]; iii+=1
    _ax.set_title("(i) TKE budget")
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

for _ax in ax[0, 1:]:
    trans = transforms.blended_transform_factory(_ax.transAxes, _ax.transData)
    _ax.plot([0, 1], [ds_ref_stat["PBLH"].to_numpy()]*2, color="pink", linestyle="-.", transform=trans)

U10_mean = np.mean(ds.U10)
V10_mean = np.mean(ds.V10)
ax[1, 0].plot(X_sT, ds.U10 - U10_mean, "k-", label="$U_{\\mathrm{10m}} - \\overline{U}_{\\mathrm{10m}}$")
ax[1, 0].plot(X_sT, ds.V10 - V10_mean, "k--",   label="$V_{\\mathrm{10m}} - \\overline{V}_{\\mathrm{10m}}$")

# SST
ax[2, 0].plot(X_sT, SST, color='blue', label="SST")
ax[2, 0].plot(X_sT, ds.T2 - zerodegC, color='red', label="$T_{\\mathrm{2m}}$")


#cbar0.ax.set_ylabel("W [$\\mathrm{cm} / \\mathrm{s}$]")
#cbar1.ax.set_ylabel("U [$\\mathrm{m} / \\mathrm{s}$]")
#cbar2.ax.set_ylabel("V [$\\mathrm{m} / \\mathrm{s}$]")



for i, _ax in enumerate(ax[0, :]):
    _ax.set_ylim(args.z_rng)
     
    if i==0:
        _ax.set_ylabel("$z$ [ m ]")
    else:
        _ax.set_ylabel("")
        _ax.set_yticks(_ax.get_yticks(), [""]*len(_ax.get_yticks()))
        _ax.grid()



ax[0, 0].set_title("(a) $W$ [$\\mathrm{cm} / \\mathrm{s}$]")
ax[2, 0].set_title("(c) SST (blue) and $T_\\mathrm{2m}$ (red)")


ax[1, 0].legend(loc="upper right")
ax[1, 0].set_ylabel("[ $ \\mathrm{m} / \\mathrm{s} $ ]", color="black")
ax[1, 0].set_title("(b) $\\left( \\overline{U}_{\\mathrm{10m}}, \\overline{V}_{\\mathrm{10m}}\\right) = \\left( %.2f, %.2f \\right)$" % (U10_mean, V10_mean,))

#ax[2, 0].legend()
ax[2, 0].set_ylabel("[ $ \\mathrm{K}$ ]", color="black")

for _ax in ax[:, 0].flatten():
    _ax.grid()
    _ax.set_xlabel("$x$ [km]")
    _ax.set_xlim(np.array(args.x_rng))



ax[1, 0].set_ylim([-2.5, 2.5])
ax[2, 0].set_ylim([13.5, 16.5])


for _ax in ax[1:, 1:].flatten():
    plt.delaxes(_ax)

if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()

