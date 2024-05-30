import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import datetime
import wrf_load_helper 
import os

def detectSmaller(x, threshold):

    x = np.array(x)

    idx = x < threshold

    result = None
    for i, _tf in enumerate(idx):
    
        if _tf:
            if np.all(idx[i:]):
                result = i
                break

    return result


def detectWithin(x, x_rng):

    x = np.array(x)
    idx = ( x >= x_rng[0] ) & ( x < x_rng[1] )
    
        
    phase = "1_finding_beg"
    beg_idx = -1
    segs = []
    for i, _tf in enumerate(idx):
        

        if phase == "1_finding_beg":
            
            if _tf:
                beg_idx = i
                phase = "2_finding_connectivity"
                continue

        elif phase == "2_finding_connectivity":
            
            if _tf:
                
                if i == len(idx) - 1:
                    
                    end_idx = i
                    segs.append([beg_idx, end_idx])

                    
            else:

                end_idx = i 
                segs.append([beg_idx, end_idx])
                phase = "1_finding_beg"

    return segs


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-dir', type=str, help='Input directory.', required=True)
parser.add_argument('--output', type=str, help='Output netcdf filename.', required=True)
parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
parser.add_argument('--PBLH-rng', type=float, nargs=2, help="Time range of PBLH", required=True)




args = parser.parse_args()

print(args)



exp_beg_time = pd.Timestamp(args.exp_beg_time)
wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)
time_beg = exp_beg_time + pd.Timedelta(hours=args.time_rng[0])
time_end = exp_beg_time + pd.Timedelta(hours=args.time_rng[1])

def relTimeInHrs(t):
    return (t - exp_beg_time.to_datetime64()) / np.timedelta64(1, 'h')

wsm = wrf_load_helper.WRFSimMetadata(
    start_datetime  = exp_beg_time,
    data_interval   = wrfout_data_interval,
    frames_per_file = args.frames_per_wrfout_file,
)

# Loading data
print("Loading wrf dir: %s" % (args.input_dir,))
ds = wrf_load_helper.loadWRFDataFromDir(
    wsm, 
    args.input_dir,
    beg_time = time_beg,
    end_time = time_end,
    prefix="analysis_",
    suffix=".nc",
    avg=None,
    verbose=False,
    inclusive="left",
)

t = relTimeInHrs(ds.coords["time"].to_numpy())
PBLH = ds["PBLH"]
LH = ds["LH"]

print(PBLH)


half_span=12

PBLH_mm = PBLH.rolling(time=2*half_span+1, center=True).mean()
LH_mm = LH.rolling(time=2*half_span+1, center=True).mean()

# Detect
segs = detectWithin(PBLH_mm.to_numpy(), args.PBLH_rng)


dlnLHdt = ( LH_mm.shift(time=-1) - LH_mm.shift(time=1) ) / (2*wrfout_data_interval.total_seconds()) / LH_mm * (86400.0)


if np.any(dlnLHdt > 0):
    print("Warning: dlnLHdt is not always negative.")


# trim the trailing NaNs
dlnLHdt = dlnLHdt.isel(time=slice(None, -(half_span+1)))


print(dlnLHdt.to_numpy())
LH_eq_idx = detectSmaller(np.abs(dlnLHdt), 0.10)

if LH_eq_idx is None:
    raise Exception("Cannot find equilibrium based on LH.")




print("Segments: ")
for i, seg in enumerate(segs):
    print("[%d] %d - %d" % (i, seg[0], seg[1],))

print("Loading matplotlib...")

import matplotlib
import matplotlib.pyplot as plt

print("Done")

fig, ax = plt.subplots(1, 1)

fig.suptitle(args.output.split('/')[-1])

ax.plot(t, PBLH, linestyle="--", color="gray")
ax.plot(t, PBLH_mm, linestyle="-",  color="black")

blended_trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)

for i, seg in enumerate(segs):
    t_beg = t[seg[0]] 
    t_end = t[seg[1]]
 
    rect = matplotlib.patches.Rectangle(
        [t_beg, 0],
        t_end - t_beg, 1,
        transform=blended_trans, edgecolor=None, facecolor=(0.9, 0.9, 0.9)
    )

    ax.add_patch(rect)

ax.plot([t[LH_eq_idx],]*2, [0, 1], "b-", transform=blended_trans)


print("Showing figures...")
plt.show()






