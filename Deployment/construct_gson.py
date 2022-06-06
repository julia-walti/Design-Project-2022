import os
import readers
import geojson


here = os.path.dirname(os.path.abspath(__file__))
gml_name = 'data/satom_calibrated_energy.gml'
gml_path = os.path.join(here, gml_name)
meteo_name = 'data/Aigle_MeteoSchweiz_2019.cli'
meteo_path = os.path.join(here, meteo_name)

print('')
gml, nrj_dem, occ_rate, appliance_rts, position = readers.read_gml(gml_path)
meteo, azimuth, zenith = readers.read_meteo(meteo_path)
data3D = readers.get_data_3D(gml, meteo, occ_rate, appliance_rts, azimuth, zenith)
data2D = readers.get_data_2D(data3D, nrj_dem)

a=data2D[data2D['building_id']==924469.0]
dico={}
for i in data2D.columns:
    if i=='energy': dico[i] = a[i][0:24].tolist()
    else: dico[i]=a[i][0:96].tolist()

energy=dico['energy']

dico['coordinates'] = position['pos'][position['id']==924469.0].tolist()
dico['type'] = 'Polygon'

#ouvre (crée) un fichier GSON
with open("input_api.gson","w") as f:
    geojson.dump(dico,f)


    
