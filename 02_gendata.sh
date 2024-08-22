#!/bin/bash

# Averge raw data to hourly data
./91_generate_hourly_avg.sh

# Make softlinks for dSST = 0.
# Different wvm but with dSST = 0 are
# actually the same outcome
./92_make_softlinks.sh

# Generate the "delta"
./93_generate_delta_analysis.sh


# Collect the analysis so that we
# can plot phase diagram: response
# as a function of dSST, wvm (L)
./94_pack_data.sh


