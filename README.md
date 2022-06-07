# Design Project 2022
## An API for the short-term simulation of heating demand in a district heating network with machine learning models

### Explication projet 
The goal of the project is to use machine learning model (here random forest regressor), to predict buildings heat demand. The model takes as input a GeoJSON file with the building characteristics, adds the heating demand predictions for 24 hours and returns the updated GeoJSON.


modele et input(details dans requirements)/output 

## Get this prediction by going to 
site

### Requirements 

## Recreate deployment
In the deployment folder, you can find:
- Deployment.py script that runs in the command prompt with the GeoJSON filepath as argument. It uploads an output file with the 24 hours prediction in the same entitie for one building.
- DeploymentV2.py script that runs in the command prompt with the GeoJSON filepath as argument. It uploads an output file with 24 entities for each hour of prediction for one building.
- App_server.py script that creates the local server where the model is deployed.
- Train_set.geojson, the 232 buildings in Monthey used to train the model.
- Test_set.geojson, the 100 buildings in Monthey used to test the model.
### Requirements
- argparse
- numpy
- pandas
- geojson
- json
- requests

## Recreate training 
In the the training folder, you can find: 
- train_test.py Script to train and test a model. The dataset can be personnalised and th model too at the /!\ CHOICES location (line 61)
- helpers.py Contains a couple helpers function, particularly the testing function
- reader.py Contains all functions to read the data, create and transform the dataset and construct the train and testing matrices
- globals.py Contains the global variables that are used in multipl scripts
- _init_.py Script to initialize and be able to use the global variables

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
