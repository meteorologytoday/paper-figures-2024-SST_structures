import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import wrf_load_helper 
import datetime

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-files', type=str, nargs='+', help='Input directories.', required=True)
parser.add_argument('--output', type=str, help='Output filename in png.', default="")
parser.add_argument('--title', type=str, help='Title', default="")
parser.add_argument('--exp-names', type=str, nargs="+", help='Title', default=None)
parser.add_argument('--exp-beg-time', type=str, help='Title', required=True)
parser.add_argument('--parameter-values', type=float, nargs="+", help="The value of parameters.", default=None)
parser.add_argument('--LH-rng', type=float, nargs=2, help="LH range", default=[20, 100])
parser.add_argument('--HFX-rng', type=float, nargs=2, help="HFX range", default=[-20, 40])
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--plot-check', action="store_true")

args = parser.parse_args()

print(args)


exp_beg_time = pd.Timestamp(args.exp_beg_time)

if args.exp_names is None:
    args.exp_names = args.input_files
else:
    if len(args.input_files) != len(args.exp_names):
        raise Exception("Error: --exp-names does not receive the same length as --input-files")

def relTimeInHrs(t):
    return (t - exp_beg_time.to_datetime64()) / np.timedelta64(1, 'h')


# Loading data
data = []
ref_ds = None
print("Start loading data.")
for i, input_file in enumerate(args.input_files):
    print("Loading file: %s" % (input_file,))

    ds = xr.open_dataset(input_file)
           

    data.append(ds)

print("Done")

plot_bundles = [

    dict(
        title="Sensible heat",
        plotting_info = [
            ["$ \\overline{C}_H \\overline{U} \\overline{T}_{OA} $", "C_H_WND_TOA",         ],
            ["$ \\overline{C'_H U'} \\overline{T}_{OA}   $",         "C_H_WND_cx_mul_TOA",  ],
            ["$ \\overline{U' T'_{OA}} \\overline{C}_{H} $",         "WND_TOA_cx_mul_C_H",  ],
            ["$ \\overline{C'_H T'_{OA}} \\overline{U}   $",         "C_H_TOA_cx_mul_WND",  ],
            ["$ \\overline{C'_H U' T'_{OA}}              $",         "C_H_WND_TOA_cx",      ],
            ["Res",                                                  "F_SEN_res",           ],
            #["HFX",                                                  "HFX",           ],

        ],
    ),

#    dict(
#        title="Latent heat",
#        plotting_info = [
#            ["$ \\overline{C}_Q \\overline{U} \\overline{Q}_{OA} $", "C_Q_WND_QOA",         ],
#            ["$ \\overline{C'_Q U'} \\overline{Q}_{OA}   $",         "C_Q_WND_cx_mul_QOA",  ],
#            ["$ \\overline{U' Q'_{OA}} \\overline{C}_{Q} $",         "WND_QOA_cx_mul_C_Q",  ],
#            ["$ \\overline{C'_Q Q'_{OA}} \\overline{U}   $",         "C_Q_QOA_cx_mul_WND",  ],
#            ["$ \\overline{C'_Q U' Q'_{OA}}              $",         "C_Q_WND_QOA_cx",      ],
#
#        ],
#    ),

]

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

nrow = 6
ncol = len(plot_bundles)

fig, ax = plt.subplots(nrow, ncol, figsize=(8, 12), squeeze=False)

for i, plot_bundle in enumerate(plot_bundles):
    
    for j, (label, varname) in enumerate(plot_bundle["plotting_info"]):
    
        _ax = ax[j, i]
        _ax.set_title("%s" % (label,))
        for k, _ds in enumerate(data):
            _y = _ds[varname]
            _x = relTimeInHrs( _ds.coords["time"].to_numpy() )
            _ax.plot(_x, _y, label=args.exp_names[k])#, color=color, linestyle=ls, label=decomp_name)
        
    #_ax.set_title("%s (ref value = %.1f)" % (plot_bundle["title"], plot_bundle["ref_val"],))
    _ax.set_ylabel("[$\\mathrm{W} / \\mathrm{m}^2$]")
    _ax.legend(loc="upper left", ncol=2)
    _ax.grid(True)

#time_fmt = "%Y/%m/%d %Hh"
#fig.suptitle("Time: %d ~ %d hr\n($\\overline{C}_H = %.1f$, $L_q \\overline{C}_Q = %.1f$, $\\overline{U} = %.1f$, $\\overline{T}_{OA} = %.1f \\mathrm{K}$, $\\overline{Q}_{OA} = %.1f \\mathrm{g}/\\mathrm{kg}$)" % (
#    relTimeInHrs(time_beg),
#    relTimeInHrs(time_end),
#    "Average: %d ~ %d km\n" % (args.x_rng[0], args.x_rng[1]) if args.x_rng is not None else "",
#    ref_C_H_m,
#    ref_C_Q_m * Lq,
#    ref_WND_m,
#    ref_TOA_m,
#    ref_QOA_m * 1e3,
#))

if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()


