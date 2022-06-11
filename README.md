# Design Project 2022
## An API for the short-term simulation of heating demand in a district heating network with machine learning models

### Explication projet 
The goal of the project is to use machine learning model (here random forest regressor), to predict buildings heat demand. The prediction is made using the building characteristics, and the weather data for the time where the prediction is asked.



## Get this prediction 
First the script App_server must be run in the command prompt to create the local server where the model is deployed.
In order to create a prediction for the heating demand of one or multiple building, the script Deployment.py (or DeploymentV2.py) needs to be run in the command prompt with the GeoJSON containing the buildings characteristics and the weather data as argument. The output file is download in the same folder.

### Requirements 
-requests
-json
-geojson
-pandas
-numpy
-argparse


## Recreate deployment
format input geojson 
In the deployment folder, you can find:
- Deployment.py script that runs in the command prompt with the GeoJSON filepath as argument. It uploads an output file with the 24 hours prediction in the same entitie for one building.
- DeploymentV2.py script that runs in the command prompt with the GeoJSON filepath as argument. It uploads an output file with 24 entities for each hour of prediction for one building.
- App_server.py script that creates the local server where the model is deployed.
- construct_json_day.py script that creates a GSON file containing weather and buildings characteristics for a chosen day.
-helpers_gson.py containing function for the predicition
-readers_gson.py containing the function to transform the GSON in usable 2D matix
### Requirements
- argparse
- numpy
- pandas
- geojson
- json
- requests
- globals.py
- -model.joblib

## Recreate training 
In the the training folder, you can find: 
- train_test.py Script to train and test a model. The dataset can be personnalised and th model too at the /!\ CHOICES location (line 61)
- helpers.py Contains a couple helpers function, particularly the testing function
- reader.py Contains all functions to read the data, create and transform the dataset and construct the train and testing matrices
- globals.py Contains the global variables that are used in multipl scripts
- _init_.py Script to initialize and be able to use the global variables
- construct_gson_day.py, the script that creates the input GSON file with the .cli weather file and the .GML CityGML2.0 file
### Requirements
- numpy
- sklearn
- joblibimport numpy 
- xml.etree.ElementTree
- pandas
- numpy
- datetime 
- pandas
- pvlib
- joblib
