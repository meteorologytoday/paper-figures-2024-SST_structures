import xarray as xr
import pandas as pd
import numpy as np
from shared_constants import *

def genAnalysis(
    ds,
    data_interval,
):

    ds = ds.mean(dim=['south_north', 'south_north_stag'], keep_attrs=True)
    
    ref_ds = ds.mean(dim=['time'], keep_attrs=True)
    Nx = ref_ds.dims['west_east']
    Nz = ref_ds.dims['bottom_top']

    X_sU = ds.DX * np.arange(Nx+1) / 1e3
    X_sT = (X_sU[1:] + X_sU[:-1]) / 2
    X_T = np.repeat(np.reshape(X_sT, (1, -1)), [Nz,], axis=0)
    X_W = np.repeat(np.reshape(X_sT, (1, -1)), [Nz+1,], axis=0)
    dX_sT = ds.DX * np.arange(Nx)

    Z_W = (ref_ds.PHB + ref_ds.PH) / 9.81
    Z_T = (Z_W[1:, :] + Z_W[:-1, :]) / 2

    ds = ds.assign_coords(dict(
        west_east = X_sT, 
        west_east_stag = X_sU, 
    ))

    merge_data = []

    # Cannot use the following to get surface pressure:
    #PRES = ds.PB + ds.P
    #SFC_PRES = PRES.isel(bottom_top=0)
    
    # This is the correct one
    SFC_PRES = ds["PSFC"]

    PRES1000hPa=1e5

    R_over_cp = 2.0 / 7.0

    dT = (np.amax(ds["TSK"].to_numpy()) - np.amin(ds["TSK"].to_numpy())) / 2 

    TA = ( 300.0 + ds["T"].isel(bottom_top=0) ).rename("TA")
    TO = (ds["TSK"] * (PRES1000hPa/SFC_PRES)**R_over_cp).rename("TO")
    TOA    = ( TO - TA ).rename("TOA")

    #  e1=svp1*exp(svp2*(tgdsa(i)-svpt0)/(tgdsa(i)-svp3)) 


    # Bolton (1980). But the formula is read from 
    # phys/physics_mmm/sf_sfclayrev.F90 Lines 281-285 (WRFV4.6.0)
    salinity_factor = 0.98
    E1 = 0.6112e3 * np.exp(17.67 * (ds["TSK"] - 273.15) / (ds["TSK"] - 29.65) ) * salinity_factor
    QSFCMR = (287/461.6) * E1 / (SFC_PRES - E1)
    
    QA  = ds["QVAPOR"].isel(bottom_top=0).rename("QA")
    QO  = QSFCMR.rename("QO")

    QOA = QO - QA
    #QOA = xr.where(QOA > 0, QOA, 0.0)
    QOA = QOA.rename("QOA")
 
    #merge_data.append(WIND10)
    merge_data.extend([TO, TA, TOA, QO, QA, QOA,])

    V_T = ds["V"]
    
    _tmp = ds["U"]
    U_T = ds["T"].copy().rename("U_T")

    U_T[:, :, :] = (_tmp.isel(west_east_stag=slice(1, None)).to_numpy() + _tmp.isel(west_east_stag=slice(0, -1)).to_numpy()) / 2
    U_T = U_T.rename("U_T")
    merge_data.append(U_T) 
 
    U_sfc = U_T.isel(bottom_top=0).to_numpy()
    V_sfc = V_T.isel(bottom_top=0)
    WND_sfc = (U_sfc**2 + V_sfc**2)**0.5
    WND_sfc = WND_sfc.rename("WND_sfc")

    HFX_from_FLHC = ds["FLHC"] * TOA
    QFX_from_FLQC = ds["FLQC"] * QOA
    LH_from_FLQC = Lq * QFX_from_FLQC

    HFX_from_FLHC = HFX_from_FLHC.rename("HFX_from_FLHC")
    QFX_from_FLQC = QFX_from_FLQC.rename("QFX_from_FLQC")
    LH_from_FLQC  = LH_from_FLQC.rename("LH_from_FLQC")

    merge_data.append(HFX_from_FLHC)
    merge_data.append(QFX_from_FLQC)
    merge_data.append(LH_from_FLQC)

    CH = ds["FLHC"] / WND_sfc
    CH = CH.rename("CH")
    
    CQ = ds["FLQC"] / WND_sfc
    CQ = CQ.rename("CQ")

    merge_data.append(WND_sfc)
    merge_data.append(CH)
    merge_data.append(CQ)


    TTL_RAIN = ds["RAINNC"] + ds["RAINC"] #+ ds["RAINSH"] + ds["SNOWNC"] + ds["HAILNC"] + ds["GRAUPELNC"]
    PRECIP = ( TTL_RAIN - TTL_RAIN.shift(time=1) ) / data_interval.total_seconds()
    
    TTL_RAIN = TTL_RAIN.rename("TTL_RAIN")
    PRECIP = PRECIP.rename("PRECIP") 


    #if "QICE_TTL" in ds:   
    #    WATER_TTL = ds["QVAPOR_TTL"] + ds["QRAIN_TTL"] + ds["QICE_TTL"] + ds["QSNOW_TTL"] + ds["QCLOUD_TTL"]
    #else:
    #    WATER_TTL = ds["QVAPOR_TTL"] + ds["QRAIN_TTL"] + ds["QCLOUD_TTL"]

    #dWATER_TTLdt = ( WATER_TTL - WATER_TTL.shift(time=1) ) / wrfout_data_interval.total_seconds()
    #dWATER_TTLdt = dWATER_TTLdt.rename("dWATER_TTLdt") 

    merge_data.append(PRECIP)
    merge_data.append(TTL_RAIN)
    #merge_data.append(dWATER_TTLdt)


    DIV10 = ( ( ds["U10"].roll(west_east=-1) - ds["U10"] ) / ds.DX ).rename("DIV10")
    VOR10 = ( ( ds["V10"].roll(west_east=-1) - ds["V10"] ) / ds.DX ).rename("VOR10")
    merge_data.append(DIV10)
    merge_data.append(VOR10)


    DIV = xr.zeros_like(ds["T"]).rename("DIV")
    tmp = ( ds["U"].roll(west_east_stag=-1) - ds["U"] ) / ds.DX
    tmp = tmp.isel(west_east_stag=slice(0, -1))
    DIV[:] = tmp[:]

    VOR = xr.zeros_like(ds["V"]).rename("VOR")
    tmp = ( ds["V"] - ds["V"].roll(west_east=1) ) / ds.DX
    tmp = (tmp.roll(west_east=-1) + tmp ) / 2.0
    VOR[:] = tmp[:]

    merge_data.append(DIV)
    merge_data.append(VOR)

    U_T = xr.zeros_like(ds["T"]).rename("U_T")
    tmp = (ds["U"].roll(west_east_stag=-1) + ds["U"] ) / 2.0
    tmp = tmp.isel(west_east_stag=slice(0, -1))
    U_T[:] = tmp[:]

    WND = ( U_T**2 + ds["V"]**2 )**0.5
    WND = WND.rename("WND")

    merge_data.append(U_T)
    merge_data.append(WND)


    new_ds = xr.merge(merge_data)

    return new_ds

