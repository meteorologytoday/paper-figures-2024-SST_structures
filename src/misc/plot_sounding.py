import xarray as xr
import pandas as pd
import numpy as np
import argparse
from WRFSoundingTools import readWRFSounding 

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input', type=str, help='Input sounding file of WRF.', required=True)
parser.add_argument('--output', type=str, help='Output filename in png.', default="")
parser.add_argument('--title', type=str, help='Title', default="")
args = parser.parse_args()

# Loading data
df_sfc, df_sdg = readWRFSounding(args.input)


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

theta = df_sdg.loc[:, 'potential_temperature']
hgt   = df_sdg.loc[:, 'height']
Q     = df_sdg.loc[:, 'mixing_ratio']
U     = df_sdg.loc[:, 'wind_x']
V     = df_sdg.loc[:, 'wind_y']

fig, ax = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'wspace': 0.4}, sharey=True)


if args.title == "":
    fig.suptitle(args.input)
else:
    fig.suptitle(args.title)

ax[0].plot(theta, hgt, 'r-')

ax0_twinx = ax[0].twiny()
ax0_twinx.plot(Q, hgt, 'b-')

ax[1].plot(U, hgt, 'k-', label='U')
ax[1].plot(V, hgt, 'k--', label='V')

ax[1].legend()


ax[0].set_xlabel(" $\\theta$ [$\\mathrm{K}$]", color='red')
ax0_twinx.set_xlabel("$q$ [$\\mathrm{kg} \\, / \\, \\mathrm{m}^3$]", color='blue')

ax[1].set_ylabel("wind speed [$\\mathrm{m} \\, / \\, \\mathrm{s}$]")

for _ax in ax.flatten():
    _ax.grid()
    _ax.set_ylabel("$z$ [$\\mathrm{m}$]")

for _ax, color, side in zip([ax[0], ax0_twinx], ['red', 'blue',], ['bottom', 'top',]):
    _ax.tick_params(color=color, labelcolor=color, axis='x')
    _ax.spines[side].set_color(color)

ax0_twinx.spines['bottom'].set_visible(False)


ax[0].set_ylim([0, 25e3])
ax[0].set_xlim([280, 500])
ax0_twinx.set_xlim([-1, 15])
ax[1].set_xlim([-20, 40])

if args.output != "":
    fig.savefig(args.output, dpi=300)

plt.show()





