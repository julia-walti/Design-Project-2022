#pip install fastapi
#pip install uvicorn[standard]
import requests
import json
import geojson
import readers_gson
import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='GeoJSON file')
parser.add_argument('GeoJSON_filepath',
                    help='GeoJSON filepath')
#parser.add_argument('heure prévision',
#                    help='hours at which we want predictions: [1,12,24] for example')

args = parser.parse_args()


with open(args.GeoJSON_filepath) as json_data:
    data_dict = json.load(json_data)
    
    
print('File opened')
print('')
    
a=data_dict['features']


#my_server = "http://127.0.0.1:8080/update"

#todo = a[0]
#response = requests.post(my_server, json=todo)

my_server = "http://127.0.0.1:8080/update"
list_feat=[ ]
print('Input file uploaded on the server ')
print(' ')
for i in range(len(a)-300):
    todo = a[i]
    response = requests.post(my_server, json=todo)
    list_feat.append(response.json())
#todo={"hello":"world"}
output={"type":"FeatureCollection","features":list_feat}

print(' ')

#output={"type":"FeatureCollection","features":[response.json()]}

with open('Demo_output.geojson', 'w') as f:
   geojson.dump(output, f)

print('Output file downloaded on the computer ')
