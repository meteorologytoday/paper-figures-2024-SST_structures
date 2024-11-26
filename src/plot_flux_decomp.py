import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import wrf_load_helper 
import datetime

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-file', type=str, help='Input file.', required=True)
parser.add_argument('--output', type=str, help='Output filename in png.', default="")
parser.add_argument('--title', type=str, help='Title', default="")
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--delta-analysis', action="store_true")
parser.add_argument('--varying-param', type=str, help='Parameters. The first and second parameters will be the varying parameters while the rest stays fixed.', required=True, choices=["dSST", "Ug", "Lx", "wnm"])
parser.add_argument('--fixed-params', type=str, nargs='*', help='Parameters that stay fixed.', required=True, choices=["dSST", "Ug", "Lx", "wnm"])
parser.add_argument('--thumbnail-numbering', type=str, help='Thumbnail numbering.', default="abcdefghijklmn")
parser.add_argument('--fixed-param-values', type=float, nargs="*", help='The values of the fixed parameters', default=[])
parser.add_argument('--ref-exp-order', type=int, help='The reference case (start from 0) to perform decomposition', default=0)
parser.add_argument('--LH-rng', type=float, nargs=2, help='The values of the LH range', default=[None, None])
parser.add_argument('--HFX-rng', type=float, nargs=2, help='The values of the HFX range', default=[None, None])
parser.add_argument('--spacing', type=float, help='The small separation between different variables', default=0.02)
parser.add_argument('--domain-size', type=float, help='The length of domain used with wnm', default=None)



args = parser.parse_args()

print(args)

plotted_fluxes = ["sensible", "latent", "momentum"]
ncols = len(plotted_fluxes)
nrows = 1
        
if args.varying_param == "wnm" and args.domain_size is None:
    raise Exception("If the varying parameter is `wnm`, `--domain-size` must be provided.")

sel_dict = {}
for i, param in enumerate(args.fixed_params):
    sel_dict[param] = args.fixed_param_values[i]

    if param == "dSST":
        sel_dict[param] /= 1e2

print("sel_dict = ", str(sel_dict))
 

print("Start loading data.")
ds = xr.open_dataset(args.input_file)#, engine="scipy")
ds = ds.sel(**sel_dict)
print(ds)

coord_x = ds.coords[args.varying_param]


if args.delta_analysis:

    plot_infos = dict(

        # Sensible
        dRHO_CH_WND_TOA = dict(
            label = "$ \\delta \\overline{\\rho}_A \\, \\overline{C}_H \\, \\overline{U}_A \\, \\overline{\\Theta}_{OA} $",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        RHO_dCH_WND_TOA = dict(
            label = "$\\overline{ \\rho }_A \\, \\delta \\overline{C}_H \\, \\overline{U}_A \\, \\overline{\\Theta}_{OA} $",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        RHO_CH_dWND_TOA = dict(
            label = "$\\overline{ \\rho }_A \\, \\overline{C}_H \\, \\delta \\overline{U}_A \\, \\overline{\\Theta}_{OA} $",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        RHO_CH_WND_dTOA = dict(
            label = "$\\overline{ \\rho }_A \\, \\overline{C}_H \\, \\overline{U}_A \\,  \\delta \\overline{\\Theta}_{OA} $",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),



        dRHO_dCH_WND_TOA = dict(
            label = "$\\overline{ \\delta \\rho \\, \\delta C_H } \\, \\overline{\\Theta}_{OA}$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        dRHO_CH_dWND_TOA = dict(
            label = "$\\overline{C}_H \\, \\overline{\\Theta}_{OA} \\, \\overline{ \\delta \\rho \\, \\delta U_A } $",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        dRHO_CH_WND_dTOA = dict(
            label = "$\\overline{C}_H \\, \\overline{U}_{A} \\, \\overline{ \\delta \\rho \\, \\delta \\Theta_{OA} } $",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        RHO_dCH_dWND_TOA = dict(
            label = "$\\overline{\\rho}_A \\, \\overline{\\Theta}_{OA} \\, \\overline{ \\delta C_H \\, \\delta U_A} $",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        RHO_dCH_WND_dTOA = dict(
            label = "$\\overline{\\rho}_A \\, \\overline{U}_A \\, \\overline{ \\delta C_H \\, \\delta \\Theta_{OA}} $",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        RHO_CH_dWND_dTOA = dict(
            label = "$\\overline{\\rho}_A \\, \\overline{C}_H \\, \\overline{ \\delta U_A \\, \\delta \\Theta_{OA}} $",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        HFX_34 = dict(
            label = "$r_\\mathrm{sen}$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        HFX_RES = dict(
            label = "$R_\\mathrm{sen}$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),


        
        # Latent
        dRHO_CQ_WND_QOA = dict(
            factor = 2.5e6,
            label = "$L_Q \\delta \\overline{\\rho}_A \\, \\overline{C}_Q \\, \\overline{U}_A \\, \\overline{Q}_{OA} $",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        RHO_dCQ_WND_QOA = dict(
            factor = 2.5e6,
            label = "$L_Q \\overline{ \\rho }_A \\, \\delta \\overline{C}_Q \\, \\overline{U}_A \\, \\overline{Q}_{OA} $",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        RHO_CQ_dWND_QOA = dict(
            factor = 2.5e6,
            label = "$L_Q \\overline{ \\rho }_A \\, \\overline{C}_Q \\, \\delta \\overline{U}_A \\, \\overline{Q}_{OA} $",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        RHO_CQ_WND_dQOA = dict(
            factor = 2.5e6,
            label = "$L_Q \\overline{ \\rho }_A \\, \\overline{C}_Q \\, \\overline{U}_A \\,  \\delta \\overline{Q}_{OA} $",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),



        dRHO_dCQ_WND_QOA = dict(
            factor = 2.5e6,
            label = "$L_Q \\overline{ \\delta \\rho \\, \\delta C_Q } \\, \\overline{Q}_{OA}$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        dRHO_CQ_dWND_QOA = dict(
            factor = 2.5e6,
            label = "$L_Q \\overline{C}_Q \\, \\overline{Q}_{OA} \\, \\overline{ \\delta \\rho \\, \\delta U_A } $",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        dRHO_CQ_WND_dQOA = dict(
            factor = 2.5e6,
            label = "$L_Q \\overline{C}_Q \\, \\overline{U}_{A} \\, \\overline{ \\delta \\rho \\, \\delta Q_{OA} } $",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        RHO_dCQ_dWND_QOA = dict(
            factor = 2.5e6,
            label = "$L_Q \\overline{\\rho}_A \\, \\overline{Q}_{OA} \\, \\overline{ \\delta C_Q \\, \\delta U_A} $",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        RHO_dCQ_WND_dQOA = dict(
            factor = 2.5e6,
            label = "$L_Q \\overline{\\rho}_A \\, \\overline{U}_A \\, \\overline{ \\delta C_Q \\, \\delta Q_{OA}} $",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        RHO_CQ_dWND_dQOA = dict(
            factor = 2.5e6,
            label = "$L_Q \\overline{\\rho}_A \\, \\overline{C}_Q \\, \\overline{ \\delta U_A \\, \\delta Q_{OA}} $",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        QFX_34 = dict(
            factor = 2.5e6,
            label = "$r_\\mathrm{lat}$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        QFX_RES = dict(
            factor = 2.5e6,
            label = "$R_\\mathrm{lat}$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),


        # Momentum
        dRHO_CD_WND2 = dict(
            label = "$\\delta \\overline{\\rho}_A \\,  \\overline{C}_D \\, \\overline{U}^2_A $",
            unit = "$ \\mathrm{kg} / \\mathrm{m} / \\mathrm{s}^2 $",
        ),

        RHO_dCD_WND2 = dict(
            label = "$\\overline{ \\rho}_A \\,  \\delta \\overline{C}_D \\, \\overline{U}^2_A $",
            unit = "$ \\mathrm{kg} / \\mathrm{m} / \\mathrm{s}^2 $",
        ),

        RHO_CD_2WNDdWND = dict(
            label = "$ 2 \\overline{ \\rho}_A \\,  \\overline{C}_D \\, \\overline{U}_A \\, \\delta \\overline{U}_A $",
            unit = "$ \\mathrm{kg} / \\mathrm{m} / \\mathrm{s}^2 $",
        ),

        dRHO_dCD_WND2 = dict(
            label = "$ \\overline{U}_A^2 \\, \\overline{ \\delta \\rho \\, \\delta C_D }$",
            unit = "$ \\mathrm{kg} / \\mathrm{m} / \\mathrm{s}^2 $",
        ),

        dRHO_CD_2WNDdWND = dict(
            label = "$ 2 \\overline{C}_D \\, \\overline{U}_A \\, \\overline{ \\delta \\rho_A \\, \\delta U_A }$",
            unit = "$ \\mathrm{kg} / \\mathrm{m} / \\mathrm{s}^2 $",
        ),

        RHO_dCD_2WNDdWND = dict(
            label = "$ 2 \\overline{\\rho}_A \\, \\overline{U}_A \\, \\overline{ \\delta C_D \\, \\delta U_A }$",
            unit = "$ \\mathrm{kg} / \\mathrm{m} / \\mathrm{s}^2 $",
        ),

        RHO_CD_dWND2 = dict(
            label = "$ \\overline{ \\rho}_A \\, \\overline{C}_D \\, \\overline{\\left( \\delta U_A \\right)^2} $",
            unit = "$ \\mathrm{kg} / \\mathrm{m} / \\mathrm{s}^2 $",
        ),

        MFX_34 = dict(
            label = "$ r_\\mathrm{mom} $",
            unit = "$ \\mathrm{kg} / \\mathrm{m} / \\mathrm{s}^2 $",
        ),

        MFX_RES = dict(
            label = "$ R_\\mathrm{mom} $",
            unit = "$ \\mathrm{kg} / \\mathrm{m} / \\mathrm{s}^2 $",
        ),


 
        dMFX_approx = dict(
            factor = 1,
            label = "$\\delta \\overline{\\tau}$",
            unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
            bracket = False,
        ),


        dHFX = dict(
            factor = 1,
            label = "$\\delta \\overline{F}_\\mathrm{sen}$",
            unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
            bracket = False,
        ),

        dLH = dict(
            factor = 1,
            label = "$\\delta \\overline{F}_\\mathrm{lat}$",
            unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
            bracket = False,
        ),

     
        dHFX_approx = dict(
            factor = 1,
            label = "Approx $\\delta \\overline{F_\\mathrm{sen}}$",
            unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
        ),

        dLH_approx = dict(
            factor = 1,
            label = "Approx $ \\delta \\overline{F_\\mathrm{lat}}$",
            unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
            
            #
        ),


    )

    """
    # Old decomposition: RHO is part of exchange coefficients

    plot_infos = dict(


        dCH_WND_TOA = dict(
            label = "$\\overline{ \\delta C_H \\, U_A \\, \\Theta_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        CH_dWND_TOA = dict(
            label = "$\\overline{ C_H \\, \\delta U_A \\, \\Theta_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        CH_WND_dTOA = dict(
            label = "$\\overline{ C_H \\, U_A \\,  \\delta \\Theta_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        CH_dWND_dTOA = dict(
            label = "$\\overline{ C_H \\, \\delta U_A \\, \\delta \\Theta_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        dCH_WND_dTOA = dict(
            label = "$\\overline{ \\delta C_H \\, U_A \\, \\delta \\Theta_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        dCH_dWND_TOA = dict(
            label = "$\\overline{ \\delta C_H \\, \\delta U_A \\, \\Theta_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        dCH_dWND_dTOA = dict(
            label = "$\\overline{ \\delta C_H \\, \\delta U_A \\, \\delta \\Theta_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),


        dCQ_WND_QOA = dict(
            factor = 2.5e6,
            label = "$\\overline{ \\delta C_Q \\, U_A \\, Q_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        CQ_dWND_QOA = dict(
            factor = 2.5e6,
            label = "$\\overline{ C_Q \\, \\delta U_A \\, Q_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        CQ_WND_dQOA = dict(
            factor = 2.5e6,
            label = "$\\overline{ C_Q \\, U_A \\,  \\delta Q_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        CQ_dWND_dQOA = dict(
            factor = 2.5e6,
            label = "$\\overline{ C_Q \\, \\delta U_A \\, \\delta Q_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),


        dCQ_WND_dQOA = dict(
            factor = 2.5e6,
            label = "$\\overline{ \\delta C_Q \\, U_A \\, \\delta Q_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        dCQ_dWND_QOA = dict(
            factor = 2.5e6,
            label = "$\\overline{ \\delta C_Q \\, \\delta U_A \\, Q_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),

        dCQ_dWND_dQOA = dict(
            factor = 2.5e6,
            label = "$\\overline{ \\delta C_Q \\, \\delta U_A \\, \\delta Q_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ),


        dHFX = dict(
            factor = 1,
            label = "$\\delta \\overline{F}_\\mathrm{sen}$",
            unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
            bracket = False,
        ),

        dLH = dict(
            factor = 1,
            label = "$\\delta \\overline{F}_\\mathrm{lat}$",
            unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
            bracket = False,
        ),

     
        dHFX_approx = dict(
            factor = 1,
            label = "Approx $\\delta \\overline{F_\\mathrm{sen}}$",
            unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
        ),

        dLH_approx = dict(
            factor = 1,
            label = "Approx $ \\delta \\overline{F_\\mathrm{lat}}$",
            unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
            
            #
        ),


    )

    """


else:

    plot_infos = dict(


        CQ_WND_QOA_cx = dict(
            factor = 2.5e6,
            label = "$L_Q \\, \\overline{ C'_Q \\, U'_A \\, Q'_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
            
        ),

        WND_QOA_cx_mul_C_Q = dict(
            factor = 2.5e6,
            label = "$L_Q \\, \\overline{C}_Q \\, \\overline{ U'_A Q'_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
            bracket = True,
        ),



        CQ_QOA_cx_mul_WND = dict(
            factor = 2.5e6,
            label = "$L_Q \\, \\overline{U}_A \\, \\overline{ C_Q' Q'_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
            
        ),


        CQ_WND_cx_mul_QOA = dict(
            factor = 2.5e6,
            label = "$L_Q \\, \\overline{Q}_{OA} \\, \\overline{ C_Q' U'_A }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
            
        ),

        CQ_WND_QOA = dict(
            factor = 2.5e6,
            label = "$L_Q \\overline{C}_Q \\, \\overline{U}_A \\, \\overline{Q}_{OA}$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
            
        ),

        CQ_WND_TOA = dict(
            factor = 1.0,
            label = "$\\overline{C}_H \\, \\overline{U}_A \\, \\overline{T}_{OA}$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
            
        ),


        WND_TOA_cx_mul_C_H = dict(
            label = "$\\overline{C}_H \\, \\overline{ U'_A T'_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
            
        ),


        CQ_WND_TOA_cx = dict(
            label = "$\\overline{ C'_H \\, U'_A \\, T'_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
            #
            
        ),

        CQ_TOA_cx_mul_WND = dict(
            label = "$\\overline{U}_A \\, \\overline{ C_H' T'_{OA} }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
            #
            
        ),


        CQ_WND_cx_mul_TOA = dict(
            label = "$\\overline{T}_{OA} \\, \\overline{ C_H' U'_A }$",
            unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
            #
            
        ),


        PRECIP = dict(
            factor = 86400.0,
            label = "Precip",
            unit = "$ \\mathrm{mm} / \\mathrm{day} $",
            
        ),


        TO = dict(
            factor = 1,
            label = "$T_O$",
            unit = "$ \\mathrm{K} $",
            
        ),

        TA = dict(
            factor = 1,
            offset = 273.15,
            label = "$\\overline{T_A}$",
            unit = "$ \\mathrm{K} $",
            
        ),

        TOA_m = dict(
            factor = 1,
            label = "$\\Theta_{OA}$",
            unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
            
        ),

        QO = dict(
            factor = 1e3,
            label = "$Q_O$",
            unit = "$ \\mathrm{g} / \\mathrm{kg} $",
            
        ),

        QA = dict(
            factor = 1e3,
            label = "$Q_A$",
            unit = "$ \\mathrm{g} / \\mathrm{kg} $",
            
        ),

        QOA_m = dict(
            factor = 1,
            label = "$Q_{OA}$",
            unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
            
        ),

        PBLH = dict(
            factor = 1,
            label = "$\\overline{H_\\mathrm{PBL}}$",
            unit = "$ \\mathrm{m} $",
            
        ),

        HFX = dict(
            factor = 1,
            label = "$\\overline{F}_\\mathrm{sen}$",
            unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
            bracket = False,
        ),

        LH = dict(
            factor = 1,
            label = "$\\overline{F}_\\mathrm{lat}$",
            unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
            bracket = False,
        ),

     
        HFX_approx = dict(
            factor = 1,
            label = "$Approx \\overline{F_\\mathrm{sen}}$",
            unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
        ),

        LH_approx = dict(
            factor = 1,
            label = "Approx $\\overline{F_\\mathrm{lat}}$",
            unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
            
            #
        ),

        WND_m = dict(
            label = "$\\overline{U} $",
            unit = "$ \\mathrm{m} / \\mathrm{s} $",
            
        ),

        CQ_m = dict(
            label = "$\\overline{C_H} $",
            unit = "$ \\mathrm{m} / \\mathrm{s} $",
        ),

    )



# =================================================================
# Figure: HFX decomposition
# =================================================================


# =================================================================
print("Loading matplotlib...")
import matplotlib
if args.no_display:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
print("Done")

import colorblind 


print("Plotting decomposition...")


figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = 6 * 0.8,
    h = 4 * 0.8,
    wspace = 1.0,
    hspace = 0.7,
    w_left = 1.0,
    w_right = 1.0,
    h_bottom = 2.0,
    h_top = 1.0,
    ncol = ncols,
    nrow = nrows,
)


fig, ax = plt.subplots(
    nrows, ncols,
    figsize=figsize,
    subplot_kw=dict(aspect="auto"),
    gridspec_kw=gridspec_kw,
    constrained_layout=False,
    squeeze=False,
    sharex=False,
)


ax_flattened = ax.flatten()


for k, flux_type in enumerate(plotted_fluxes):


    _ax = ax_flattened[k]


    if args.delta_analysis:

        # When rho is included in exchange coefficients
        varnames_styles = dict(
 
            sensible_full = [

                "dHFX",
                "dRHO_CH_WND_TOA",
                "RHO_dCH_WND_TOA",
                "RHO_CH_dWND_TOA",
                "RHO_CH_WND_dTOA",
             
                "dRHO_dCH_WND_TOA",
                "dRHO_CH_dWND_TOA",
                "dRHO_CH_WND_dTOA",
                "RHO_dCH_dWND_TOA",
                "RHO_dCH_WND_dTOA",
                "RHO_CH_dWND_dTOA",

                "HFX_34",
     
            ],

            latent_full   = [

                "dLH",
                
                "dRHO_CQ_WND_QOA",
                "RHO_dCQ_WND_QOA",
                "RHO_CQ_dWND_QOA",
                "RHO_CQ_WND_dQOA",
             
                "dRHO_dCQ_WND_QOA",
                "dRHO_CQ_dWND_QOA",
                "dRHO_CQ_WND_dQOA",
                "RHO_dCQ_dWND_QOA",
                "RHO_dCQ_WND_dQOA",
                "RHO_CQ_dWND_dQOA",

                "QFX_34",
            ],

            momentum_full = [

                "dMFX_approx",
                "dRHO_CD_WND2",
                "RHO_dCD_WND2",
                "RHO_CD_2WNDdWND",
                
                "dRHO_dCD_WND2",
                "dRHO_CD_2WNDdWND",
                "RHO_dCD_2WNDdWND",
                "RHO_CD_dWND2",
                
                "MFX_34",
           ], 
           
           sensible = [

                ( "dHFX", "black", "-" ),
                ( "dRHO_CH_WND_TOA", "bluishgreen", "-" ),
                ( "RHO_dCH_WND_TOA", "skyblue", "-" ),
                ( "RHO_CH_dWND_TOA", "orange", "-" ),
                ( "RHO_CH_WND_dTOA", "reddishpurple", "-" ),
             
                ( "RHO_dCH_dWND_TOA", "bluishgreen", "--" ),
                ( "RHO_dCH_WND_dTOA", "orange", "--" ),
                ( "RHO_CH_dWND_dTOA", "skyblue", "--" ),

                ( "HFX_RES", "yellow", ":"),
     
            ],

            latent   = [

                ( "dLH", "black", "-" ),
                
                ( "dRHO_CQ_WND_QOA", "bluishgreen", "-" ),
                ( "RHO_dCQ_WND_QOA", "skyblue", "-" ),
                ( "RHO_CQ_dWND_QOA", "orange", "-" ),
                ( "RHO_CQ_WND_dQOA", "reddishpurple", "-" ),
             
                ( "RHO_dCQ_dWND_QOA", "bluishgreen", "--" ),
                ( "RHO_dCQ_WND_dQOA", "orange", "--" ),
                ( "RHO_CQ_dWND_dQOA", "skyblue", "--" ),

                ( "QFX_RES", "yellow", ":" ),
            ],

            momentum = [

                ( "dMFX_approx", "black", "-" ),
                ( "dRHO_CD_WND2", "bluishgreen", "-" ),
                ( "RHO_dCD_WND2", "skyblue", "-" ),
                ( "RHO_CD_2WNDdWND", "reddishpurple", "-" ),
                
                ( "RHO_dCD_2WNDdWND", "orange", "--" ),
                ( "RHO_CD_dWND2", "skyblue", "--" ),
                
                ( "MFX_RES", "yellow", ":"),
           ] 
        )[flux_type]

        """
        # When rho is included in exchange coefficients
        varnames = dict(
            
            sensible = [
                "dHFX",
                "dCH_WND_TOA",
                "CH_dWND_TOA",
                "CH_WND_dTOA",
                "CH_dWND_dTOA",
                "dCH_WND_dTOA",
                "dCH_dWND_TOA",
                "dCH_dWND_dTOA",
            ],

            latent   = [
                "dLH",
                "dCQ_WND_QOA",
                "CQ_dWND_QOA",
                "CQ_WND_dQOA",
                "CQ_dWND_dQOA",
                "dCQ_WND_dQOA",
                "dCQ_dWND_QOA",
                "dCQ_dWND_dQOA",
            ],
            
        )[flux_type]
        """

    else:
        varnames = dict(
            Sensible = ["HFX", "CQ_WND_TOA", "WND_TOA_cx_mul_C_H", "CQ_TOA_cx_mul_WND", "CQ_WND_cx_mul_TOA"],
            Latent   = ["LH",  "CQ_WND_QOA", "WND_QOA_cx_mul_C_Q", "CQ_QOA_cx_mul_WND", "CQ_WND_cx_mul_QOA"],
        )[flux_type]

    if args.varying_param == "dSST":
        title_param = "$\\Delta \\mathrm{SST}$"
    elif args.varying_param == "Ug":
        title_param = "$U_g$"
    elif args.varying_param == "Lx":
        title_param = "$L$"
    elif args.varying_param == "wnm":
        title_param = "$L$"


    vartext = dict(
        sensible = "$\\delta \\overline{F}_\\mathrm{sen}$",
        latent   = "$\\delta \\overline{F}_\\mathrm{lat}$",
        momentum = "$\\delta \\overline{\\tau}$",
    )
 
    _ax.set_title(
        "(%s) %s as a function of %s" % (
            args.thumbnail_numbering[k],
            vartext[flux_type],
            title_param,
        ),
        size=20,
    )
    for i, (varname, color, linestyle) in enumerate(varnames_styles):


        color = colorblind.BW8color[color]

        _plot_info = plot_infos[varname]
        
        factor  = _plot_info["factor"] if "factor" in _plot_info else 1.0
        offset  = _plot_info["offset"] if "offset" in _plot_info else 0.0
        bracket = _plot_info["bracket"] if "bracket" in _plot_info else True

        _plot_data = (ds[varname] + offset) * factor


        _ref_m = 0.0# _plot_data.sel(stat="mean")[0].to_numpy()
        
        d_m = _plot_data.sel(stat="mean") - _ref_m
        d_s = _plot_data.sel(stat="std")

        if args.varying_param == "dSST":
            _coord_x = coord_x + args.spacing * i
        elif args.varying_param == "Lx":    
            _coord_x = coord_x + args.spacing * i
            #pblh = ds["PBLH"].sel(stat='mean')
            #_coord_x = ( pblh.to_numpy()**2 / (coord_x.to_numpy() * 1e3) )
            #_coord_x = 1 / coord_x.to_numpy()

            #print("COORD_X = ", _coord_x)
    
        elif args.varying_param == "wnm":
            _coord_x = args.domain_size / coord_x + args.spacing * i


        if args.delta_analysis:

            label = _plot_info["label"]

        else:

            if bracket:
                var_label = "$\\delta ($%s$)$" % (_plot_info["label"],)
            else:    
                var_label = "$\\delta$%s" % (_plot_info["label"],)

            label = "%s (%.2f %s)" % (var_label, _ref_m, _plot_info["unit"])



            
        _ax.errorbar(_coord_x, d_m, yerr=d_s, fmt='o-', markersize=6, capsize=5, color=color, linewidth=1.5, elinewidth=1.5, linestyle=linestyle, label=label)

        _ax.set_ylabel("[ %s ] " % (_plot_info["unit"]), size=15)

        _ax.grid(visible=True)


    xlabel_text = ""
    if args.varying_param == "dSST":
        xlabel_text = "$\\Delta \\mathrm{SST}$ [ $\\mathrm{K}$ ]"
    elif args.varying_param == "Ug":
        xlabel_text = "$U_\\mathrm{g}$ [ $\\mathrm{m} \\, / \\, \\mathrm{s}$ ]"
    elif args.varying_param == "Lx":
        xlabel_text = "$ L $ [ $\\mathrm{km} $ ]"
    elif args.varying_param == "wnm":
        xlabel_text = "$ L $ [ $\\mathrm{km} $ ]"


    _ax.set_xlabel(xlabel_text, size=15)
        #_ax.set_xlabel("$ H^2_{\\mathrm{pbl}} / L $ [ $\\mathrm{m} $ ]")
        #_ax.set_xlabel("$ L^{-1} $ [ $\\mathrm{km}^{-1} $ ]")


    """
    if flux_type == "Sensible":
        _ax.set_ylim(args.HFX_rng)
    elif flux_type == "Latent":
        _ax.set_ylim(args.LH_rng)
    """


    _ax.legend(loc="upper center", ncols=3, mode="expand", bbox_to_anchor=(0., -0.25, 1., .102))




if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()



