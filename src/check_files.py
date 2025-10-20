import xarray as xr
import pandas as pd
import numpy as np
import argparse
import wrf_load_helper 
from pathlib import Path

import datetime
import sys


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-root',  type=str, help='Input directories.', required=True)
    parser.add_argument('--data-type',   type=str, help='Can be: `raw` or `preavg`. ', choices=["raw", "preavg"], required=True)
    parser.add_argument('--output-log-dir', type=str, help='Output filename for not-ok simulations', default=".")
    parser.add_argument('--U',           type=int, help='Backaground wind.', required=True)
    args = parser.parse_args()

    print(args)

    print("This is a hard-coded check for all the pre-avg data.")
    print("If there is any data that does not pass check, the message will be output to stderr.")
    

    output_log_file = Path(args.output_log_dir) / datetime.datetime.now().strftime(f"data_detect_{args.data_type:s}_%Y-%m-%d_%H%M%S.log")

    
    
    data = None

    case_infos = []

    exp_beg_time = pd.Timestamp("2001-01-01T00:00:00")

    check_time_beg = exp_beg_time
    check_time_end = exp_beg_time + pd.Timedelta(days=15)

    print("IMPORTANT -- Output log file: ", output_log_file)
    with open(output_log_file, "w") as log_file:
        
        def log(*args, **kwargs):
            print(*args, flush=True, file=log_file, **kwargs)

        for target_lab in ["FULL", "SIMPLE"]:
            for bl_scheme in ["MYNN25", "MYJ", "YSU"]:
                for U in [args.U, ]:
                    for wnm in [0, 4, 5, 7, 10, 20, 40]:
                        
                        dTs = dict(
                            FULL   = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                            SIMPLE = [0.0 , 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                        )[target_lab]

                        for dT in dTs:

                            if target_lab == "FULL":
         
                                if wnm == 0 and dT == 0:
                                    pass
                                elif (dT == 0) != (wnm == 0):
                                    continue
                                elif not ( wnm in [10, ] or dT in [1,] ):
                                    continue
       
                            elif target_lab == "SIMPLE":

                                if wnm == 0 and dT == 0:
                                    pass
                                elif (dT == 0) != (wnm == 0):
                                    continue
                                elif (wnm, dT) not in [ (5, 1.0) , (10, 1.0) ]:
                                    continue
                                
                            mph = "on" if target_lab == "FULL" else "off"
                            dT_int = int(dT*100)
                            casedir = str(Path(args.input_root) / f"lab_{target_lab:s}" / f"case_mph-{mph:s}_wnm{wnm:03d}_U{U:02d}_dT{dT_int:03d}_{bl_scheme:s}")

                            case_infos.append(dict(
                                casedir = casedir,
                                target_lab = target_lab,
                                bl_scheme = bl_scheme,
                                U = U,
                                wnm = wnm,
                                dT = dT,
                                mph = mph,
                            ))

                            print(str(casedir))


        if len(case_infos) != ( 14 + 3 ) * 3:
            raise Exception(f"Number of cases count is not correct. There should be {(14+3)*3:d} cases. We have {len(case_infos):d} cases here.")

        for case_info in case_infos:
            
            target_lab = case_info["target_lab"]
            bl_scheme = case_info["bl_scheme"]
            U = case_info["U"]
            wnm = case_info["wnm"]
            dT  = case_info["dT"]
            mph = case_info["mph"]
            casedir = case_info["casedir"]
            
            error_msgs = []         
            if args.data_type == "raw": 
                wrfout_data_interval = pd.Timedelta(seconds=600)
            elif args.data_type == "preavg":
                wrfout_data_interval = pd.Timedelta(hours=1)

            wsm = wrf_load_helper.WRFSimMetadata(
                start_datetime  = exp_beg_time,
                data_interval   = wrfout_data_interval,
                frames_per_file = 0,
            )


            ds = None
            try:
                print("Loading wrf dir: %s" % (casedir,))
                ds = wrf_load_helper.loadWRFDataFromDir(
                    wsm, 
                    casedir,
                    beg_time = check_time_beg,
                    end_time = check_time_end,
                    prefix="wrfout_d01_",
                    suffix="",
                    avg=None,
                    verbose=False,
                    inclusive="both",
                )

                t = ds.coords["time"].to_numpy() # datetime64
                if t[0] != pd.to_datetime(check_time_beg):
                    error_msgs.append(f"Begin time {str(check_time_beg):s} data is not detected")
         
                if t[-1] != pd.to_datetime(check_time_end):
                    error_msgs.append(f"End time {str(check_time_end):s} data is not detected")
                
                dt = t[1:] - t[:-1]
                if not np.all( dt == pd.to_timedelta(wrfout_data_interval) ):
                    error_msgs.append(f"Data intervals are not consistent in simulation folder  `{str(casedir):s}`")

            except Exception as e:
                
                import traceback
                error_msgs.append(f"Exception happens in casedir=`{str(casedir):s}`: {str(e):s}")
                traceback.print_exc()
           


            if len(error_msgs) == 0:
                error_case_info = None
            else: 
                error_case_info = dict(
                    casedir = casedir,
                    error_msgs = error_msgs,    
                )
                
                log(f"# Case: {str(casedir):s}")
                for i, error_msg in enumerate(error_msgs):
                    log(f"{i+1:d}: {error_msg:s}")

                log("")
            


             












