import os
import readers
import geojson
import numpy as np

#21 mars est le 80eme jour de la semaine donc commence à l'heure 1 896 la 7 eme heure 1903

gml_name = 'satom_calibrated_energy.gml'
meteo_name = 'Aigle_MeteoSchweiz_2019.cli'

print('')
gml, nrj_dem, occ_rate, appliance_rts, position = readers.read_gml(gml_name)
meteo, azimuth, zenith = readers.read_meteo(meteo_name)
data3D = readers.get_data_3D(gml, meteo, occ_rate, appliance_rts, azimuth, zenith)
data2D = readers.get_data_2D(data3D, nrj_dem)
liste=[]
n=0
for ID in gml['id']:
    a=data2D[data2D['building_id']==ID]
    #constrcution of the 'properties' part
    dico={}
    meteo_col=['appliance', 'occ_rate', 'hour', 'we?', 'month', 'G_dh', 'G_h', 'Ta', 'Ts', 'FF', 'DD', 'RH', 'RR', 'azimuth', 'zenith']
    for i in data2D.columns:
        if i=='energy': dico[i] = a[i][1903-24:1903].tolist()      #choose the hour before the day where we want predictions
        elif i in meteo_col : dico[i]=a[i][1903:1903+96].tolist()  # meteo forecast for the hours where we make predictions
        else : dico[i]= np.asarray(a[i])[1]
            
            
    #pour calculer les erreurs du modèles
    
    
    #construction of the 'geometry' part
    dico['coordinates']= position['pos'][n].tolist()
    for i in range(len(dico['coordinates'])):
        dico['coordinates'][i]=[float(j) for j in position['pos'][n].tolist()[i]]
    dico['coordinates']=[dico['coordinates']]
    pos={'coordinates':dico['coordinates']}
    del dico['coordinates']

    typ={'type' : 'Polygon'}
    geom=typ|pos
    
    
    body1={"type":"Feature","geometry":geom,"properties":dico}
    liste.append(body1)  
    n=n+1

    
output={"type":"FeatureCollection","features":liste}

#ouvre (crée) un fichier GSON
with open("input_demo.gson","w") as f:
    geojson.dump(output,f)
