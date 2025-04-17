import xarray as xr
import numpy as np
import os
import os.path
import re
from datetime import datetime

default_Z_levs = [300, 500, 900]


def interpolateZ(ds, varnames, z, T_grid_var="T"):
   

    Z_W = ( (ds.PHB + ds.PH) / 9.81 ).to_numpy()
    Z_T = ds[T_grid_var].copy().rename("Z_T")
    Z_T.data[:] = (Z_W[:, 1:, :] + Z_W[:, :-1, :]) / 2

    west_east = None
    if "west_east" in ds.coords:
        west_east = ds.coords["west_east"]

    p = np.sort(np.array(z, dtype=np.float64))
    
    test_dim = ('time', 'bottom_top', 'west_east', )

    if Z_T.dims != test_dim:
        print(Z_T.dims)
        raise Exception('Error: dimension has to be time, z, x, y on T-grid.')
   
    # Check variables
    for varname in varnames:
        da = ds[varname]

        if da.dims != test_dim:
            print(Z_T.dims)
            raise Exception("Error: %s's dimension has to be (time, z, y, x) on T-grid." % (varname,))
   

    new_data_vars = []

    Nt = da.sizes['time']
    Neta = da.sizes['bottom_top']
    Nx = da.sizes['west_east']
    Nz = len(z)

    rearr_Z = np.moveaxis(Z_T.to_numpy(), [0, 1, 2], [2, 0, 1]).reshape((Neta, -1))

    da_z = xr.DataArray(
        data=z,
        dims=["Z",],
        coords=dict(
            Z=(["Z",], z),
        ),
    
        attrs=dict(
            description="Z.",
            units="hPa",
        ),
    )

    for varname in varnames:
    
        print("Interpolating variable `%s`" % (varname,))

        da = ds[varname] 

        original_data = da.to_numpy()
        rearr_data = np.moveaxis(original_data, [0, 1, 2], [2, 0, 1]).reshape((Neta, -1))

        new_data = np.zeros( (Nt*Nz*Nx,), dtype=original_data.dtype ).reshape((Nz, -1))
        
        for i in range(rearr_data.shape[1]):
            new_data[:, i] = np.interp(z, rearr_Z[:, i], rearr_data[:, i], left=np.nan, right=np.nan)
           
        # Transform the axis back 
        new_data = np.moveaxis(new_data.reshape((Nz, Nx, Nt)), [2, 0, 1], [0, 1, 2])
 
        coords = dict(
            Z = da_z,
        )

        if west_east is not None:
            coords["west_east"] = ( ["west_east", ], west_east.to_numpy())
       
        new_data_vars.append(xr.DataArray(
            data = new_data,
            dims = ["time", "Z", "west_east"],
            coords = coords,
        ).rename(varname))
    
    
    new_ds = xr.merge(new_data_vars)
    
    return new_ds



