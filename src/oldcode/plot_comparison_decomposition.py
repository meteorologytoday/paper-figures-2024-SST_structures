import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import wrf_load_helper 
import datetime

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-files', type=str, nargs='+', help='Input directories.', required=True)
parser.add_argument('--labels', type=str, nargs="+", help='Title', default=None)
parser.add_argument('--output', type=str, help='Output filename in png.', required=True)
parser.add_argument('--title', type=str, help='Title', default="")
parser.add_argument('--decomp-type', type=str, help='Title', choices=["HFX", "LH"], required=True)
parser.add_argument('--no-display', action="store_true")
args = parser.parse_args()

print(args)

if args.labels is None:
    args.labels = args.input_files
else:
    if len(args.input_files) != len(args.labels):

        print(args.input_files)
        print(args.labels)

        raise Exception("Error: --labels does not receive the same length as --input-files")


data = []
print("Start loading data.")
for i, input_file in enumerate(args.input_files):
    print("Loading file: %s" % (input_file,))

    data.append(xr.open_dataset(input_file))

print("Done")

# =================================================================
print("Loading matplotlib...")
import matplotlib
if args.no_display:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
print("Done")


print("Plotting decomposition...")


if args.decomp_type == "HFX":
    plotting_list = [
        ["$ \\overline{U}  \\Delta \\left( \\overline{ C'_H T'_{OA} }\\right) $", "WND_d(C_H_TOA_cx)",  ("dodgerblue", "-")],
        ["$ \\overline{C}_H  \\Delta \\left( \\overline{ U' T'_{OA}}\\right) $", "C_H_d(WND_TOA_cx)",  ("dodgerblue", "-")],

    ]
elif args.decomp_type == "LH":
    plotting_list = [
        ["$ \\overline{U}  \\Delta \\left( \\overline{ C'_Q Q'_{OA} }\\right) $", "WND_d(C_Q_QOA_cx)",  ("dodgerblue", "-")],
        ["$ \\overline{C}_Q  \\Delta \\left( \\overline{ U' Q'_{OA}}\\right) $", "C_Q_d(WND_QOA_cx)",  ("dodgerblue", "-")],

    ]

fig, ax = plt.subplots(len(plotting_list), 1, figsize=(8, 12), sharex=True)

for i, plotting_var in enumerate(plotting_list):
    
    _ax = ax[i]
    decomp_name, varname, (color, ls) = plotting_var

    for j, _ds in enumerate(data):
    
        _x = _ds.coords["dT"]
        _y = _ds[varname]

        _ax.scatter(_x, _y, s=20)
        _ax.plot(_x, _y, linestyle=ls, label=args.labels[j])
        
    _ax.set_ylabel("[$\\mathrm{W} / \\mathrm{m}^2$]")
    #_ax.legend(loc="upper left", ncol=2)
    _ax.legend()
    _ax.grid(True)
    _ax.set_title(decomp_name)
    _ax.set_xlabel("Amplitude [ K ]")

fig.suptitle(args.title)



if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()



