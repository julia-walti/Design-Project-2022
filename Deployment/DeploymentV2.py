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
#parser.add_argument('n_hours',
#                    help='nb d heure de predictions')
#args = parser.parse_args()


with open(args.GeoJSON_filepath) as json_data:
    data_dict = json.load(json_data)
    
    
print('File opened')
print('')
    
a=data_dict['features']


#my_server = "http://127.0.0.1:8080/update"



my_server = "http://127.0.0.1:8080/update"
list_feat=[ ]
print('Input file uploaded on the server ')
print(' ')

label_pred=['pred1','pred2','pred3','pred4','pred5','pred6','pred7','pred8','pred9','pred10','pred11','pred12','pred13','pred14','pred15','pred16','pred17','pred18','pred19','pred20','pred21','pred22','pred23','pred24']

for i in range(len(a)):
    todo = a[i]
    response = requests.post(my_server, json=todo)
    for j in label_pred:
        resp_json= response.json()

        for n in label_pred:
            if n==j:
                resp_json['properties']['PREDICTION']= resp_json['properties'][n] 
                del resp_json['properties'][n]
            else:
                del resp_json['properties'][n]
        list_feat.append(resp_json)
#todo={"hello":"world"}
output={"type":"FeatureCollection","features":list_feat}

print(' ')


with open('Demo_output.geojson', 'w') as f:
   geojson.dump(output, f)

print('Output file downloaded on the computer ')
