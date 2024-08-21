import numpy as np
import xarray as xr
import os

import traceback
from pathlib import Path

import argparse



# Bolton (1980)
# Find specific humidity using temperature, pressure, and relative humidity
def T2q(T, P, RH): # T in Kelvin, results in Pa
    E1 = 0.6112e3 * np.exp(17.67 * (T - 273.15) / (T - 29.65) ) * RH
    q = 0.622 * E1 / (P - E1)
    return q


parser = argparse.ArgumentParser(
                    prog = "collect_flux_analysis_SQ15.py",
                    description = 'This program collect the necessary information from each SQ15 solution and bundle them into a file of parameter space (DTheta, Ug, Lx, dSST)',
)

parser.add_argument('--input-dir', type=str, help='Input directory that contains all cases', required=True)

parser.add_argument('--RHs', type=float, nargs='+', help='Relative humidity', required=True)
parser.add_argument('--DThetas', type=float,   nargs='+', help='dSSTs in K.', required=True)
parser.add_argument('--Ugs',    type=float, nargs='+', help='Selected Ugs.', required=True)
parser.add_argument('--wvlens', type=int,   nargs='+', help='Wave lengths in km.', required=True)
parser.add_argument('--dSSTs',  type=float, nargs='+', help='Selected dSSTs.', required=True)
parser.add_argument('--output', type=str, help='Output netcdf file', required=True)

args = parser.parse_args()
print(args)

pathlist = []
sorting = []

varnames = ["UTOA", "UQOA"]

RHs     = args.RHs
Ugs     = args.Ugs
Lxs     = args.wvlens
dSSTs   = args.dSSTs
DThetas = args.DThetas


data = xr.Dataset(

    data_vars = {
        k : ( ["RH", "DTheta", "Ug", "Lx", "dSST"], np.zeros((len(RHs), len(DThetas), len(Ugs), len(Lxs), len(dSSTs)))) 
        for k in varnames
    },

    coords=dict(
        RH=(["RH", ], RHs),
        DTheta=(["DTheta", ], DThetas),
        Ug=(["Ug", ], Ugs),
        Lx=(["Lx", ], Lxs),
        dSST=(["dSST", ], dSSTs),
    ),
)


                
for i, RH in enumerate(RHs):
    for j, DTheta in enumerate(DThetas):
        for k, Ug in enumerate(Ugs):
            for l, Lx in enumerate(Lxs):
                for m, dSST in enumerate(dSSTs):

                    filename = os.path.join(
                        args.input_dir,
                        "DTheta_{DTheta:.1f}-Ug_{Ug:.1f}-wvlen_{Lx:03d}-dSST_{dSST:.2f}.nc".format(
                            DTheta = DTheta,
                            Ug = Ug,
                            Lx = Lx,
                            dSST = dSST,
                        ),
                    )

                    print("Loading case: ", filename)

                    _ds = xr.open_dataset(filename)
                    _ds = _ds.isel(y_T=0)
                   
                    # This is just a check
                    if _ds.attrs["dSST"] != dSST:
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
                    data["UTOA"][i,j,k,l,m] = np.sum(dx_T * U * TOA) / np.sum(dx_T)

                        
                    sst_total = (_ds.attrs["Theta0"] + _ds["sst_1"]).to_numpy()
                    pt_total  = (_ds.attrs["Theta0"] + _ds["pt_1"]).to_numpy()

                    Qsfc = T2q(sst_total, 1000e2, 1.0)
                    Qair = T2q(pt_total,  1000e2, RH)

                    QOA = Qsfc - Qair
                    data["UQOA"][i,j,k,l,m] = np.sum(dx_T * U * QOA) / np.sum(dx_T)
                 


print("Output file: ", args.output)
data.to_netcdf(args.output)
