import traceback
from multiprocessing import Pool
import multiprocessing
from pathlib import Path
import os

import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import wrf_load_helper 
import datetime
import os
from shared_constants import *



def onlyPos(x):
    return 0.0 if x < 0.0 else x

def genAnalysis(
    input_dir,
    input_dir_base,
    output_filename,
    exp_beg_time,
    wrfout_data_interval,
    frames_per_wrfout_file,
    reltime_rngs,
    avg_before_analysis,
    analysis_style,
):
    
    result = dict(output_filename=output_filename, status='UNKNOWN')

    try:
    
        exp_beg_time = pd.Timestamp(exp_beg_time)
    
        merge_data = []


        for i, reltime_rng in enumerate(reltime_rngs):
            
            selected_time_beg = exp_beg_time + reltime_rng[0]
            selected_time_end = exp_beg_time + reltime_rng[1]

            print("[%d] Analyzing time inverval: %s ~ %s" % (
                i,
                selected_time_beg.strftime("%Y-%m-%d_%H:%M:%S"), 
                selected_time_end.strftime("%Y-%m-%d_%H:%M:%S"),           
            ))
             
            ds = genAnalysis_subset(
                input_dir,
                input_dir_base,
                exp_beg_time,
                wrfout_data_interval,
                frames_per_wrfout_file,
                reltime_rng,
                avg_before_analysis,
                analysis_style,
            )
            

            merge_data.append(ds)



        print("Merging data...")
        new_ds = xr.merge(merge_data)

            
        full_range_time_beg = exp_beg_time + reltime_rngs[0][0]
        full_range_time_end = exp_beg_time + reltime_rngs[-1][1]
        new_ds.attrs["time_beg"] = full_range_time_beg.strftime("%Y-%m-%d_%H:%M:%S")
        new_ds.attrs["time_end"] = full_range_time_end.strftime("%Y-%m-%d_%H:%M:%S")
        new_ds.attrs["analysis_style"] = analysis_style 

        print("Writing file: %s" % (output_filename,))
        new_ds.to_netcdf(
            output_filename,
            unlimited_dims="time",
            encoding={'time':{'units':'hours since 2001-01-01'}}
        )

        
        result['status'] = 'OK'

    
    except Exception as e:
        result['status'] = 'ERROR'
        traceback.print_exc()

    return result


def preprocessing(
    ds,
    data_interval,
):

    ds = ds.mean(dim=['south_north', 'south_north_stag'], keep_attrs=True)
    
    ref_ds = ds.mean(dim=['time'], keep_attrs=True)
    Nx = ref_ds.dims['west_east']
    Nz = ref_ds.dims['bottom_top']

    X_sU = ref_ds.DX * np.arange(Nx+1) / 1e3
    X_sT = (X_sU[1:] + X_sU[:-1]) / 2
    X_T = np.repeat(np.reshape(X_sT, (1, -1)), [Nz,], axis=0)
    X_W = np.repeat(np.reshape(X_sT, (1, -1)), [Nz+1,], axis=0)

    Z_W = (ref_ds.PHB + ref_ds.PH) / 9.81
    Z_T = (Z_W[1:, :] + Z_W[:-1, :]) / 2

    ds = ds.assign_coords(dict(
        west_east = X_sT, 
        west_east_stag = X_sU, 
    ))

    merge_data = [ds,]

    # Cannot use the following to get surface pressure:
    #PRES = ds.PB + ds.P
    #SFC_PRES = PRES.isel(bottom_top=0)
    
    # This is the correct one
    SFC_PRES = ds["PSFC"]

    PRES1000hPa=1e5

    R_over_cp = 2.0 / 7.0

    dT = (np.amax(ds["TSK"].to_numpy()) - np.amin(ds["TSK"].to_numpy())) / 2 

    TA = ( 300.0 + ds["T"].isel(bottom_top=0) ).rename("TA")
    TOA    = ( ds["TSK"] * (PRES1000hPa/SFC_PRES)**R_over_cp - TA ).rename("TOA")

    #  e1=svp1*exp(svp2*(tgdsa(i)-svpt0)/(tgdsa(i)-svp3)) 


    # Bolton (1980). But the formula is read from 
    # phys/physics_mmm/sf_sfclayrev.F90 Lines 281-285 (WRFV4.6.0)
    salinity_factor = 0.98
    E1 = 0.6112e3 * np.exp(17.67 * (ds["TSK"] - 273.15) / (ds["TSK"] - 29.65) ) * salinity_factor
    QSFCMR = (287/461.6) * E1 / (SFC_PRES - E1)
    
    QA  = ds["QVAPOR"].isel(bottom_top=0).rename("QA")

    QOA = QSFCMR - ds["QVAPOR"].isel(bottom_top=0)
    #QOA = xr.where(QOA > 0, QOA, 0.0)
    QOA = QOA.rename("QOA")
 
    #merge_data.append(WIND10)
    merge_data.append(TOA)
    merge_data.append(QOA)

    #merge_data.append( ( (ds["TSK"] - ds["T2"]) * WIND10 ).rename("WIND10TAO") )
 
    V_T = ds["V"]
    
    _tmp = ds["U"]
    U_T = ds["T"].copy().rename("U_T")

    #print("Dimesion of U_T: ")
    #print(U_T)

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



   
    new_ds = xr.merge(merge_data)

    new_ds.attrs["dT"] = dT

    return new_ds


# This decomposition is the simple one
def analysisStyle1(
    ds, ds_base,
):

    def diffVar(varname, newname=""):
        da = ds[varname] - ds_base[varname]

        if newname != "":
            da = da.rename(newname)
        
        return da


    dCH = diffVar("CH")
    dCQ = diffVar("CQ")
    dWND = diffVar("WND_sfc")
    dTOA = diffVar("TOA")
    dQOA = diffVar("QOA")
   
    dCH_WND_TOA = (dCH * ds_base["WND_sfc"] * ds_base["TOA"]).mean(dim="west_east").rename("dCH_WND_TOA")
    CH_dWND_TOA = (ds_base["CH"] * dWND * ds_base["TOA"]).mean(dim="west_east").rename("CH_dWND_TOA")
    CH_WND_dTOA = (ds_base["CH"] * ds_base["WND_sfc"] * dTOA).mean(dim="west_east").rename("CH_WND_dTOA")

    CH_dWND_dTOA = (ds_base["CH"] * dWND * dTOA).mean(dim="west_east").rename("CH_dWND_dTOA")
    dCH_WND_dTOA = (dCH * ds_base["WND_sfc"] * dTOA).mean(dim="west_east").rename("dCH_WND_dTOA")
    dCH_dWND_TOA = (dCH * dWND * ds_base["TOA"]).mean(dim="west_east").rename("dCH_dWND_TOA")

    dCH_dWND_dTOA = (dCH * dWND * dTOA).mean(dim="west_east").rename("dCH_dWND_dTOA")
    
    
    dCQ_WND_QOA = (dCQ * ds_base["WND_sfc"] * ds_base["QOA"]).mean(dim="west_east").rename("dCQ_WND_QOA")
    CQ_dWND_QOA = (ds_base["CQ"] * dWND * ds_base["QOA"]).mean(dim="west_east").rename("CQ_dWND_QOA")
    CQ_WND_dQOA = (ds_base["CQ"] * ds_base["WND_sfc"] * dQOA).mean(dim="west_east").rename("CQ_WND_dQOA")
    
    dCQ_dWND_QOA = (dCQ * dWND * ds_base["QOA"]).mean(dim="west_east").rename("dCQ_dWND_QOA")
    CQ_dWND_dQOA = (ds_base["CQ"] * dWND * dQOA).mean(dim="west_east").rename("CQ_dWND_dQOA")
    dCQ_WND_dQOA = (dCQ * ds_base["WND_sfc"] * dQOA).mean(dim="west_east").rename("dCQ_WND_dQOA")
    
    dCQ_dWND_dQOA = (dCQ * dWND * dQOA).mean(dim="west_east").rename("dCQ_dWND_dQOA")
    
    
    dHFX_approx = (

          dCH_WND_TOA
        + CH_dWND_TOA
        + CH_WND_dTOA

        + CH_dWND_dTOA
        + dCH_WND_dTOA
        + dCH_dWND_TOA

        + dCH_dWND_dTOA
    
    ).rename("dHFX_approx")


    dQFX_approx = (
        
          dCQ_WND_QOA
        + CQ_dWND_QOA
        + CQ_WND_dQOA

        + CQ_dWND_dQOA 
        + dCQ_WND_dQOA 
        + dCQ_dWND_QOA 

        + dCQ_dWND_dQOA

    ).rename("dQFX_approx")

    dLH_approx = (Lq * dQFX_approx).rename("dLH_approx")
    
    dHFX = diffVar("HFX", "dHFX").mean(dim="west_east")
    dQFX = diffVar("QFX", "dQFX").mean(dim="west_east")
    dLH  = diffVar("LH", "dLH").mean(dim="west_east")
    
    dHFX_from_FLHC = diffVar("HFX_from_FLHC", "dHFX_from_FLHC").mean(dim="west_east")
    dQFX_from_FLQC = diffVar("QFX_from_FLQC", "dQFX_from_FLQC").mean(dim="west_east")
    dLH_from_FLQC = (dQFX_from_FLQC * Lq).rename("dLH_from_FLQC")
   
 
    #HFX  = ds["HFX"].mean(dim="west_east")
    #HFX_approx  = ( ds["CH"] * ds["WND_sfc"] * ds["TOA"] ).mean(dim="west_east").rename("HFX_approx")
    #HFX_from_FLHC  = ds["HFX_from_FLHC"].mean(dim="west_east")
 
    merge_data = []
        
    merge_data.extend([
        dHFX_approx, dQFX_approx, dLH_approx,
        dHFX, dQFX, dLH,
        dHFX_from_FLHC, dQFX_from_FLQC, dLH_from_FLQC,

        #dHFX_approx2,

        dCH_WND_TOA,
        CH_dWND_TOA,
        CH_WND_dTOA,

        dCH_dWND_TOA,
        CH_dWND_dTOA,
        dCH_WND_dTOA,
        dCH_dWND_dTOA,
        
 
        dCQ_WND_QOA,
        CQ_dWND_QOA,
        CQ_WND_dQOA,       
 
        dCQ_dWND_QOA,
        CQ_dWND_dQOA,
        dCQ_WND_dQOA,
        dCQ_dWND_dQOA,
 
    ])

    return merge_data 
    
    

# This decomposition includes the spatial ones
def analysisStyle2(
    ds, ds_base,
):

    ds_base_m = ds_base.mean(dim="west_east")

    def diffVar(varname, newname=""):
        da = ds[varname] - ds_base[varname]

        if newname != "":
            da = da.rename(newname)
        
        return da

    def horDecomp(da, name_m="mean", name_p="prime"):
        m = da.mean(dim="west_east").rename(name_m)
        p = (da - m).rename(name_p) 
        return m, p


    dCH = diffVar("CH")
    dCQ = diffVar("CQ")
    dWND = diffVar("WND_sfc")
    dTOA = diffVar("TOA")
    dQOA = diffVar("QOA")
 
    dCHm, dCHp = horDecomp(dCH)
    dCQm, dCQp = horDecomp(dCQ)
    dWNDm, dWNDp = horDecomp(dWND)
    dTOAm, dTOAp = horDecomp(dTOA)
    dQOAm, dQOAp = horDecomp(dQOA)
   
    #print(ds_base["WND_sfc"])


    dCHm_WNDm_TOAm = (dCHm * ds_base_m["WND_sfc"] * ds_base_m["TOA"]).rename("dCHm_WNDm_TOAm")
    CHm_dWNDm_TOAm = (ds_base_m["CH"] * dWNDm * ds_base_m["TOA"]).rename("CHm_dWNDm_TOAm")
    CHm_WNDm_dTOAm = (ds_base_m["CH"] * ds_base_m["WND_sfc"] * dTOAm).rename("CHm_WNDm_dTOAm")

    CHm_dWNDp_dTOAp = ( ds_base_m["CH"] * (dWNDp * dTOAp).mean(dim="west_east")     ).rename("CHm_dWNDp_dTOAp")
    WNDm_dCHp_dTOAp = ( ds_base_m["WND_sfc"] * (dCHp * dTOAp).mean(dim="west_east") ).rename("WNDm_dCHp_dTOAp")
    TOAm_dCHp_dWNDp = ( ds_base_m["TOA"] * (dCHp * dWNDp).mean(dim="west_east")     ).rename("TOAm_dCHp_dWNDp")

    CHm_dWNDm_dTOAm = (ds_base_m["CH"] * dWNDm * dTOAm).rename("CHm_dWNDm_dTOAm")
    WNDm_dCHm_dTOAm = (ds_base_m["WND_sfc"] * dCHm * dTOAm).rename("WNDm_dCHm_dTOAm")
    TOAm_dCHm_dWNDm = (ds_base_m["TOA"] * dCHm * dWNDm).rename("TOAm_dCHm_dWNDm")

    dCH_dWND_dTOA = (dCH * dWND * dTOA).mean(dim="west_east").rename("dCH_dWND_dTOA")
    
    RES_H = (CHm_dWNDm_dTOAm + WNDm_dCHm_dTOAm + TOAm_dCHm_dWNDm + dCH_dWND_dTOA).rename("RES_H")

    #===========
 
    dCQm_WNDm_QOAm = (dCQm * ds_base_m["WND_sfc"] * ds_base_m["QOA"]).rename("dCQm_WNDm_QOAm")
    CQm_dWNDm_QOAm = (ds_base_m["CQ"] * dWNDm * ds_base_m["QOA"]).rename("CQm_dWNDm_QOAm")
    CQm_WNDm_dQOAm = (ds_base_m["CQ"] * ds_base_m["WND_sfc"] * dQOAm).rename("CQm_WNDm_dQOAm")

    CQm_dWNDp_dQOAp = ( ds_base_m["CQ"] * (dWNDp * dQOAp).mean(dim="west_east")     ).rename("CQm_dWNDp_dQOAp")
    WNDm_dCQp_dQOAp = ( ds_base_m["WND_sfc"] * (dCQp * dQOAp).mean(dim="west_east") ).rename("WNDm_dCQp_dQOAp")
    QOAm_dCQp_dWNDp = ( ds_base_m["QOA"] * (dCQp * dWNDp).mean(dim="west_east")     ).rename("QOAm_dCQp_dWNDp")

    CQm_dWNDm_dQOAm = (ds_base_m["CQ"] * dWNDm * dQOAm).rename("CQm_dWNDm_dQOAm")
    WNDm_dCQm_dQOAm = (ds_base_m["WND_sfc"] * dCQm * dQOAm).rename("WNDm_dCQm_dQOAm")
    QOAm_dCQm_dWNDm = (ds_base_m["QOA"] * dCQm * dWNDm).rename("QOAm_dCQm_dWNDm")

    dCQ_dWND_dQOA = (dCQ * dWND * dQOA).mean(dim="west_east").rename("dCQ_dWND_dQOA")
    
    RES_Q = (CQm_dWNDm_dQOAm + WNDm_dCQm_dQOAm + QOAm_dCQm_dWNDm + dCQ_dWND_dQOA).rename("RES_Q")

    
    dHFX_approx = (

          dCHm_WNDm_TOAm
        + CHm_dWNDm_TOAm
        + CHm_WNDm_dTOAm

        + CHm_dWNDp_dTOAp
        + WNDm_dCHp_dTOAp
        + TOAm_dCHp_dWNDp

        + RES_H
    
    ).rename("dHFX_approx")


    dQFX_approx = (
 
          dCQm_WNDm_QOAm
        + CQm_dWNDm_QOAm
        + CQm_WNDm_dQOAm

        + CQm_dWNDp_dQOAp
        + WNDm_dCQp_dQOAp
        + QOAm_dCQp_dWNDp

        + RES_Q

    ).rename("dQFX_approx")

    dLH_approx = (Lq * dQFX_approx).rename("dLH_approx")
    
    dHFX = diffVar("HFX", "dHFX").mean(dim="west_east")
    dQFX = diffVar("QFX", "dQFX").mean(dim="west_east")
    dLH  = diffVar("LH", "dLH").mean(dim="west_east")
    
    dHFX_from_FLHC = diffVar("HFX_from_FLHC", "dHFX_from_FLHC").mean(dim="west_east")
    dQFX_from_FLQC = diffVar("QFX_from_FLQC", "dQFX_from_FLQC").mean(dim="west_east")
    dLH_from_FLQC = (dQFX_from_FLQC * Lq).rename("dLH_from_FLQC")
   
    merge_data = []
        
    merge_data.extend([
        dHFX_approx, dQFX_approx, dLH_approx,
        dHFX, dQFX, dLH,
        dHFX_from_FLHC, dQFX_from_FLQC, dLH_from_FLQC,

        dCHm_WNDm_TOAm,
        CHm_dWNDm_TOAm,
        CHm_WNDm_dTOAm,

        CHm_dWNDp_dTOAp,
        WNDm_dCHp_dTOAp,
        TOAm_dCHp_dWNDp,
        
        CHm_dWNDm_dTOAm,
        WNDm_dCHm_dTOAm,
        TOAm_dCHm_dWNDm,

        dCH_dWND_dTOA,
    
        RES_H,

        #######

        dCQm_WNDm_QOAm,
        CQm_dWNDm_QOAm,
        CQm_WNDm_dQOAm,

        CQm_dWNDp_dQOAp,
        WNDm_dCQp_dQOAp,
        QOAm_dCQp_dWNDp,
        
        CQm_dWNDm_dQOAm,
        WNDm_dCQm_dQOAm,
        QOAm_dCQm_dWNDm,

        dCQ_dWND_dQOA,
    
        RES_Q,
 
    ])

    return merge_data


def genAnalysis_subset(
    input_dir,
    input_dir_base,
    exp_beg_time,
    wrfout_data_interval,
    frames_per_wrfout_file,
    reltime_rng,
    avg_before_analysis,
    analysis_style,
):



    time_beg = exp_beg_time + reltime_rng[0]
    time_end = exp_beg_time + reltime_rng[1]
    
    time_beg_str = time_beg.strftime("%Y-%m-%dT%H:%M:%S")
    time_end_str = time_end.strftime("%Y-%m-%dT%H:%M:%S")

    time_bnd = xr.DataArray(
        name="time_bnd",
        data=[time_beg, time_end],
        dims=["time_bnd",],
        coords=dict(
            
        ),
        attrs=dict(
            reference_time=pd.Timestamp("1970-01-01"),
        ),
    )

    def relTimeInHrs(t):
        return (t - exp_beg_time.to_datetime64()) / np.timedelta64(1, 'h')


    data_interval = pd.Timedelta(seconds=wrfout_data_interval)

    # Loading data
        
    print("Loading directories: %s , and %s" % (input_dir, input_dir_base))

    wsm = wrf_load_helper.WRFSimMetadata(
        start_datetime = exp_beg_time,
        data_interval = data_interval,
        frames_per_file = frames_per_wrfout_file,
    )

    ds = wrf_load_helper.loadWRFDataFromDir(
        wsm, 
        input_dir,
        beg_time = time_beg,
        end_time = time_end,
        prefix="wrfout_d01_",
        avg="ALL" if avg_before_analysis else None,
        verbose=False,
        inclusive="both",
    )


    ds_base = wrf_load_helper.loadWRFDataFromDir(
        wsm, 
        input_dir_base,
        beg_time = time_beg,
        end_time = time_end,
        prefix="wrfout_d01_",
        avg="ALL" if avg_before_analysis else None,
        verbose=False,
        inclusive="both",
    )


    #print("Processing...")

    ds      = preprocessing(ds, data_interval)
    ds_base = preprocessing(ds_base, data_interval)


    if analysis_style == "STYLE1":
        
        merge_data = analysisStyle1(ds, ds_base)
   
    elif analysis_style == "STYLE2": 
        
        merge_data = analysisStyle2(ds, ds_base)

    else:
    
        raise Exception("Unknown analysis_style = `%s`" % (analysis_style,))



    # Adding extra variables
    for varname in [
        "IWV", "IVT", "IVT_x", "IVT_y", "TTL_RAIN", "PRECIP", 
    ]:
        
        diff_da = ds[varname] - ds_base[varname] 
        merge_data.append(diff_da) 


    # Merging data
    new_ds = xr.merge(merge_data)
  
    if avg_before_analysis is False:
        new_ds = new_ds.mean(dim="time", skipna=True, keep_attrs=True).expand_dims(
            dim = {"time": [time_beg,]},
            axis = 0,
        )
        
    new_ds.attrs["dT"] = ds.attrs["dT"]
    new_ds.attrs["time_beg"] = time_beg_str
    new_ds.attrs["time_end"] = time_end_str

    new_ds = new_ds.compute()

    return new_ds


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-dir', type=str, help='Input directories.', required=True)
    parser.add_argument('--input-dir-base', type=str, help='Input directories.', required=True)
    parser.add_argument('--output-dir', type=str, help='Output directory.', required=True)
    parser.add_argument('--exp-beg-time', type=str, help='Experiment begin time.', required=True)
    parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
    parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)

    parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
    parser.add_argument('--time-avg-interval', type=int, help="The interval of time to do the average. Unit is in minutes. If it is not specified or 0 then all data is averaged.", default=0)
    parser.add_argument('--output-count', type=int, help="The numbers of output in a file.", default=1)
    parser.add_argument('--avg-before-analysis', type=str, help="If set true, then the program will average first before analysis. This might affect the correlation terms.", choices=["TRUE", "FALSE"], required=True)
    parser.add_argument('--nproc', type=int, help="Number of parallel CPU.", default=1)
    parser.add_argument('--analysis-style', type=str, help=".", choices=["STYLE1", "STYLE2"], required=True)
    
    args = parser.parse_args()

    print(args)
   
    avg_before_analysis = True if args.avg_before_analysis == "TRUE" else False 
  

    # Decide the time decomposition
    print("Processing ...")

    exp_beg_time = pd.Timestamp(args.exp_beg_time)
    reltime_beg = pd.Timedelta(hours=args.time_rng[0])
    reltime_end = pd.Timedelta(hours=args.time_rng[1])
    time_avg_interval = pd.Timedelta(minutes=args.time_avg_interval)
    



    if time_avg_interval / pd.Timedelta(seconds=1) == 0:  # if not specified or 0
        print("The parameter `--time-avg-interval` is zero, assume the average is the entire interval.")
        time_avg_interval = time_end - time_beg
   
    print("Create dir: %s" % (args.output_dir,))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

 

    input_args = []
    
    number_of_intervals = (reltime_end - reltime_beg) / time_avg_interval
    if number_of_intervals % 1 != 0:
        raise Exception("Error: the interval of time_rng is not a multiple of time_avg_interval. Ratio: %.f" % (number_of_intervals,)) 

    number_of_intervals = int( np.ceil(number_of_intervals) ) 
    
    number_of_output_files = number_of_intervals / args.output_count
    if number_of_intervals % 1 != 0:
        raise Exception("Error: the number_of_intervals = %d is not a multiple of number_of_output_files = %d " % (
            number_of_intervals,
            number_of_output_files,
        ))
    number_of_output_files = int(number_of_output_files)
    
    print("There will be %d output files. Each contains %d data points" % (number_of_output_files, args.output_count,))
    
    time_interval_per_file = time_avg_interval * args.output_count
    for i in range(number_of_output_files):

        reltime_rngs = []
            
        reltime_beg_of_file = reltime_beg + i * time_interval_per_file
        time_beg_of_file = exp_beg_time + reltime_beg_of_file

        for j in range(args.output_count):
            
            reltime_rngs.append(
                [ 
                    reltime_beg_of_file + j     * time_avg_interval,
                    reltime_beg_of_file + (j+1) * time_avg_interval,
                ]
            )

        filename = os.path.join(
            args.output_dir,
            "analysis_{timestr:s}.nc".format(
                timestr = time_beg_of_file.strftime("%Y-%m-%d_%H:%M:%S"),
            )
        )

        if os.path.exists(filename):
            print("File %s already exists. Skip it." % (filename,))
            continue

       
        input_args.append(
            (
                args.input_dir,
                args.input_dir_base,
                filename,
                args.exp_beg_time,
                args.wrfout_data_interval,
                args.frames_per_wrfout_file,
                reltime_rngs,
                avg_before_analysis,
                args.analysis_style,
            )
        )
    
    
    
    failed_files = []
    with Pool(processes=args.nproc) as pool:

        results = pool.starmap(genAnalysis, input_args)

        for i, result in enumerate(results):
            if result['status'] != 'OK':
                print('!!! Failed to generate output : %s.' % (result['output_filename'],))
                failed_files.append(result['output_filename'])


    print("Tasks finished.")

    print("Failed files: ")
    for i, failed_file in enumerate(failed_files):
        print("%d : %s" % (i+1, failed_file,))


    print("Done")
    
    
    
    
