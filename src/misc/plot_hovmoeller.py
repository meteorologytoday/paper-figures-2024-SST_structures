import xarray as xr
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--casedir', type=str, help='Input directory.', required=True)
parser.add_argument('--output', type=str, help='Output filename in png.', default="")
parser.add_argument('--title', type=str, help='Title', default="")
args = parser.parse_args()

casedir = args.casedir
data_file = "wrfout_d01_0001-01-01_00:00:00"
idx_z = 3
stat_xrng = np.array([200, 500])

# Loading data
data_file_full = "%s/%s" % (casedir, data_file)
print("Loading file: %s" % (data_file_full,))
ds = xr.open_dataset(data_file_full, engine='scipy')

#times = pd.Timestamp(["%s %s" % (t[0:10], t[11:19]) for t in ds.Times.astype(str).to_numpy()])
times = [np.datetime64(pd.Timestamp("%s %s" % (t[0:10], t[11:19]))) for t in ds.Times.astype(str).to_numpy()]

times = np.array(times)


Nx = ds.dims['west_east']
Nz = ds.dims['bottom_top']

# Eta value on T grid
eta_T = ds.ZNU[0, :]
eta_W = ds.ZNW[0, :]

print("Selected idx_z=%d  ->  eta = %f" % (idx_z, eta_T[idx_z]))


X = ds.DX * np.arange(Nx) / 1e3


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


fig, ax = plt.subplots(1, 2, figsize=(12, 6))

fig.suptitle(casedir)

W = np.mean(ds.W[:, idx_z:idx_z+1, 0, :], axis=1)  * 1e2
U = ds.U[:, idx_z, 0, :]
U = (U[:, :-1] + U[:, 1:]) / 2

print(W.shape)
print(U.shape)
print(X.shape)
print(times.shape)

w_levs = np.linspace(-5, 5, 21) / 10
u_levs = np.linspace(8, 12, 21)

mappable1 = ax[0].contourf(X, times, W, levels=w_levs, cmap="bwr", extend="both")
mappable2 = ax[1].contourf(X, times, U, levels=u_levs, cmap="jet", extend="both")


cbar1 = plt.colorbar(mappable1, ax=ax[0], orientation="vertical")
cbar2 = plt.colorbar(mappable2, ax=ax[1], orientation="vertical")

ax[0].set_title("W")
ax[1].set_title("U")

cbar1.ax.set_label("[$\\times 10^{-2} \\, \\mathrm{m} / \\mathrm{s}$]")
cbar2.ax.set_label("[$\\mathrm{m} / \\mathrm{s}$]")

for _ax in ax.flatten():
    _ax.grid()
    _ax.set_xlabel("[km]")


if args.output != "":
    fig.savefig(args.output, dpi=300)

plt.show()





