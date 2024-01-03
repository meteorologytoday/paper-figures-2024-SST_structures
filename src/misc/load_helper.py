import xarray as xr
import pandas as pd
import numpy as np

def loadWRFData(filename):
    
    ds = xr.open_dataset(filename, decode_times=False, engine='scipy')
    t = [pd.Timestamp("%s %s" % (t[0:10], t[11:19])) for t in ds.Times.astype(str).to_numpy()]
   
    ts = xr.DataArray(
        data = t,
        dims = ["Time"],
    ).rename('time')
  
    ds = xr.merge([ds, ts]).rename({'Time':'time'})

    return ds

