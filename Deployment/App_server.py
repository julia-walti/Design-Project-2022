import cherrypy
import simplejson
import readers_gson
import helpers_gson
import globals_quali as globals
import joblib
import os
import json
import numpy as np
class Root(object):

    @cherrypy.expose
    def update(self):
        cl = cherrypy.request.headers['Content-Length']
        rawbody = cherrypy.request.body.read(int(cl))
        body = simplejson.loads(rawbody)
        
        globals.prev = 0
        globals.n_hours=24
        globals.t_prev = []
        
        globals.nb_h = 96
        globals.nb_bd = 1
         
        to_drop = [] 
        to_prev = ['occ_rate','G_dh','G_h','Ta','Ts','FF','DD','RH', 'appliance']
        
        here = os.path.dirname(os.path.abspath(__file__))
        path_model = os.path.join(here, 'model_tpot3_50.joblib') #le modèle est sur le serveur donc on peut le loader comme çca?
        model=joblib.load(path_model)

       
        data, energy_prev, position = readers_gson.get_data(body)
        
        # Transform data
        data_init = readers_gson.transform(data, energy_prev, to_drop = to_drop, prev = globals.prev, to_prev = to_prev)
        data_use = data_init.drop(['building_id'], axis=1, inplace=False)
        # Prediction
        pred = helpers_gson.prediction(data_use, model)
        
        
        # Create output
         #constrcution of the 'properties' part
        for n in range(len(pred)):
            propert={}
            hourly=['appliance', 'occ_rate', 'hour',
           'we?', 'month', 'G_dh', 'G_h', 'Ta', 'Ts', 'FF', 'DD', 'RH', 'RR',
           'azimuth', 'zenith']
            for i in data_init.columns:
                if i in hourly: propert[i]= data_init[i].tolist()
                else : propert[i]=data_init[i][0]

            pos={'coordinates':position}
            typ={'type' : 'Polygon'}
            geom=typ|pos

            #propert =data_init.to_dict()
            label_pred =['pred1','pred2','pred3','pred4','pred5','pred6','pred7','pred8','pred9','pred10','pred11','pred12','pred13','pred14','pred15','pred16','pred17','pred18','pred19','pred20','pred21','pred22','pred23','pred24']
            x=0
            for i in label_pred:
                propert[i] = pred[x]
                x=x+1

            body1={"type":"Feature","geometry":geom,"properties":propert}
        #body['coordinates'] = position
        #body.insert(1, "coordinates", position, allow_duplicates=False)
        
        json_object = json.dumps(body1)

        # do_something_with(body)
        return json_object

    @cherrypy.expose
    def index(self):
        return """
<html>
<script type="text/javascript" src="http://ajax.googleapis.com/ajax/libs/jquery/1.4.2/jquery.min.js"></script>
<script type='text/javascript'>
function Update() {
    $.ajax({
      type: 'POST',
      url: "update",
      contentType: "application/json",
      processData: false,
      data: $('#updatebox').val(),
      success: function(data) {alert(data);},
      dataType: "text"
    });
}
</script>
<body>
<input type='textbox' id='updatebox' value='{}' size='20' />
<input type='submit' value='Update' onClick='Update(); return false' />
</body>
</html>
"""

cherrypy.quickstart(Root())