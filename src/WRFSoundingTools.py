import numpy as np
import pandas as pd

def readWRFSounding(filename):
    
    # sfc = surface, sdg = sounding
    surface_air_pressure, surface_potential_temperature, surface_mixing_ratio = np.loadtxt(filename, usecols=(0,1,2), unpack=True)

    # Only need the first record
    surface_air_pressure = surface_air_pressure[0:1]
    surface_potential_temperature = surface_potential_temperature[0:1]
    surface_mixing_ratio = surface_mixing_ratio[0:1]
    
    height, potential_temperature, mixing_ratio, wind_x, wind_y = np.loadtxt(filename, skiprows=1, unpack=True)
    
    sfc_dict = dict(
        air_pressure=surface_air_pressure,
        potential_temperature = surface_potential_temperature,
        mixing_ratio=surface_mixing_ratio,
    )
    
    
    sdg_dict = dict(
        height = height,
        potential_temperature = potential_temperature,
        mixing_ratio = mixing_ratio,
        wind_x = wind_x,
        wind_y = wind_y,
    )
    
    df_sfc = pd.DataFrame(data=sfc_dict)
    df_sdg = pd.DataFrame(data=sdg_dict)

    return df_sfc, df_sdg

def writeWRFSounding(filename, df_sfc, df_sdg):

    with open(filename, 'w') as f:
        
        f.write('%10.5f\t%10.5f\t%10.5f\n' % (
            df_sfc.at[0, 'air_pressure'],
            df_sfc.at[0, 'potential_temperature'],
            df_sfc.at[0, 'mixing_ratio'],
        ));
        
 
    with open(filename, 'a') as f:
        
        for i in range(df_sdg.shape[0]):
            f.write('%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f\n' % (
                df_sdg.at[i, 'height'],
                df_sdg.at[i, 'potential_temperature'],
                df_sdg.at[i, 'mixing_ratio'],
                df_sdg.at[i, 'wind_x'],
                df_sdg.at[i, 'wind_y'],
            ));
    
