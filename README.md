# Description
This is the code to generate the figures of the paper "Examining the SST Tendency Forced by Atmospheric Rivers" submitted to XXX journal.

# Prerequisite

1. Python >= 3.7
    - Matplotlib
    - Cartopy
    - Numpy
    - Scipy
    - netCDF4
2. ImageMagick >= 6.9.10

# Reproducing Figures

1. Clone this project.
2. Download the file `data.zip` from [XXX](http://XXX)
3. Unzip the folder `data` from `data.zip` into this git project root folder (i.e., the same folder containing this `README.md` file).
4. Run `00_runall.sh`.
5. The figures are generated in the folder `final_figures`.







# Notes

## WRF's phase diagram of flux analysis

- Run `51_generate_flux_analysis.sh`
- Run `61_collect_flux.sh` runs to generate nc files
- Run `71_plot_WRF_phase.sh` to plot.


