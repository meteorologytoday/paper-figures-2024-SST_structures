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

parser.add_argument('--input-dir', type=str, help='Input directory that contains all cases', required=True)
parser.add_argument('--selected-dSST', type=float, help='Selected fixed dSST', required=True)
parser.add_argument('--selected-DTheta', type=float, help='Selected fixed DTheta', required=True)
parser.add_argument('--RH', type=float, help='Relative humidity', required=True)
parser.add_argument('--Ugs',    type=float, nargs='+', help='Selected Ugs.', required=True)
parser.add_argument('--wvlens', type=int,   nargs='+', help='Wave lengths in km.', required=True)
parser.add_argument('--output', type=str, help='Output file', default="")
parser.add_argument('--no-display', action="store_true")

args = parser.parse_args()
print(args)

pathlist = []
sorting = []

varnames = ["UTOA", "UQOA"]

Ugs = args.Ugs
wvlens = args.wvlens
dSSTs = np.array([args.selected_dSST])
DThetas = np.array([args.selected_DTheta])


data = xr.Dataset(

    data_vars = {
        k : ( ["DTheta", "Ug", "wvlen", "dSST"], np.zeros((len(DThetas), len(Ugs), len(wvlens), len(dSSTs)))) 
        for k in varnames
    },

    coords=dict(
        DTheta=(["DTheta", ], DThetas),
        Ug=(["Ug", ], Ugs),
        wvlen=(["wvlen", ], wvlens),
        dSST=(["dSST", ], dSSTs),
    ),
)

for i, Ug in enumerate(args.Ugs):
    for j, wvlen in enumerate(args.wvlens):
        
        filename = os.path.join(
            args.input_dir,
            "DTheta_{DTheta:.1f}-Ug_{Ug:.1f}-wvlen_{wvlen:03d}-dSST_{dSST:.2f}.nc".format(
                DTheta = args.selected_DTheta,
                Ug = Ug,
                wvlen = wvlen,
                dSST = args.selected_dSST,
            ),
        )

        print("Loading case: ", filename)

        _ds = xr.open_dataset(filename)
        _ds = _ds.isel(y_T=0)
       
        if _ds.attrs["dSST"] != args.selected_dSST:
            raise Exception("Problem loading dSST! Not consistent with selected dSST")

        
        # Do integration
        dx_T = _ds["dx_T"].to_numpy()
       
        U = ( (_ds["u_0"] + _ds["u_1"])**2 + (_ds["v_0"] + _ds["v_1"])**2 )**0.5
        U = U.isel(z_T=0).to_numpy() 
        TOA = (_ds["sst_1"] -  _ds["pt_1"]).to_numpy()
     
        #U_1 = ((_ds["u_1"]**2 + _ds["v_1"]**2)**0.5).isel(z_T=0).to_numpy()
        #U_1 = _ds["u_1"].isel(z_T=0).to_numpy()
        #U_0 = U_1*0 + (( _ds["u_0"]**2 + _ds["v_0"]**2 )**0.5).isel(z_T=0).to_numpy()

        #data["U1delta1"][i] = np.sum(dx_T * U_1 * delta_1) / np.sum(dx_T)
        #data["U0delta1"][i] = np.sum(dx_T * U_0 * delta_1) / np.sum(dx_T)
        U_mean = np.sum(dx_T * U) / np.sum(dx_T)
        TOA_mean = np.sum(dx_T * TOA) / np.sum(dx_T)

        #data["Udelta_FULL"][i, j, 0] = np.sum(dx_T * U * delta) / np.sum(dx_T)
        #data["U_mean"][i, j, 0]      = U_mean
        #data["delta_mean"][i, j, 0]  = delta_mean
        #data["Udelta_CORR"][i, j, 0] = np.sum(dx_T * (U - U_mean) * (delta - delta_mean)) / np.sum(dx_T)
        data["UTOA"][0, i, j, 0] = np.sum(dx_T * U * TOA) / np.sum(dx_T)

        # Bolton (1980)
        def T2q(T, P, RH): # T in Kelvin, results in Pa
            E1 = 0.6112e3 * np.exp(17.67 * (T - 273.15) / (T - 29.65) ) * RH
            q = 0.622 * E1 / (P - E1)
            return q
            
        sst_total = (_ds.attrs["Theta0"] + _ds["sst_1"]).to_numpy()
        pt_total  = (_ds.attrs["Theta0"] + _ds["pt_1"]).to_numpy()

        Qsfc = T2q(sst_total, 1000e2, 1.0)
        Qair = T2q(pt_total,  1000e2, args.RH)

        QOA = Qsfc - Qair
        data["UQOA"][0, i, j, 0] = np.sum(dx_T * U * QOA) / np.sum(dx_T)
     


        
"""
const_H = 6.9
ref_idx = 0
data_dif = dict(
    WND = np.zeros((N,)),
    THM = np.zeros((N,)),
    COR = np.zeros((N,)),
    FUL = np.zeros((N,)),
)

for i, path in enumerate(pathlist):
    
    data_dif["WND"][i] = const_H * (data["U_mean"][i] - data["U_mean"][ref_idx]) * data["delta_mean"][ref_idx]
    data_dif["THM"][i] = const_H * (data["delta_mean"][i] - data["delta_mean"][ref_idx]) * data["U_mean"][ref_idx]
    data_dif["COR"][i] = const_H * (data["Udelta_CORR"][i] - data["Udelta_CORR"][ref_idx])
    data_dif["FUL"][i] = const_H * (data["Udelta_FULL"][i] - data["Udelta_FULL"][ref_idx])

data_dif["RES"] = data_dif["FUL"] - (data_dif["THM"] + data_dif["COR"] + data_dif["WND"])
"""

print("Domain size: sum(dx_T) = ", np.sum(dx_T))


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

fig.suptitle("$ \\Delta \\Theta = {DTheta:d} \\mathrm{{K}} $, $ \\mathrm{{RH}} = {RH:.1f} \\% $".format(
    DTheta = int(args.selected_DTheta),
    RH     = args.RH * 1e2,
))
 
for i, varname in enumerate(varnames):
    
    _ax = ax.flatten()[i]

    plot_info = plot_infos[varname]

    _plot_data = data[varname].isel(dSST=0).isel(DTheta=0).to_numpy() / plot_info['factor']
    _cntr_levs = plot_info['cntr_levs']
    cs = _ax.contour(wvlens, Ugs, _plot_data, _cntr_levs, colors='black')
    plt.clabel(cs)

    _ax.set_title("{label:s} [{unit:s}]".format(
        label = plot_info["label"],
        unit  = plot_info["unit"],
    ))

    _ax.set_xlabel("$ L_x $ [ km ]")
    _ax.set_ylabel("$ U_\\mathrm{g} $ [ m / s ]")


for _ax in ax.flatten():
    _ax.grid()

if not args.no_display:
    plt.show()

if args.output != "":

    print("Saving output: ", args.output)    
    fig.savefig(args.output, dpi=200)


