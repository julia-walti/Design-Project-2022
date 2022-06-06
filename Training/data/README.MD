# README

Please keep these files confidential.
Do not share them, do not load them into any git repository (even private).

## satom_calibrated_energy.gml

The file `satom_calibrated_energy.gml` contains the CityGML model for 332 buildings whose CitySim-simulated energy demand was calibrated by varying the following parameters:

- Window U-Value
- Thermal bridges (psi)
- Infiltration rate (ninf)
- Minimimum indoor temperature (tmin)

Only buildings with the following characteristics have been retained:

- Absolute Relative Error of final cumulated energy < 2%
- Mean-Normalized RMSE < 50 %
- calibrated using data for 2019

The building `gml_id` corresponds to the EGID from the RegBL.

Please note that at the moment no context (terrain, other buildings) is provided in the file.

## Aigle_MeteoSchweiz_2019.cli

This hourly weather file for 2019 contains weather data from MeteoSwiss for the closest weather station (Aigle VD). It was used to run the simulation.



