import numpy as np
from helpers import angle_azimuth
from lxml import etree
import xml.etree.ElementTree as ET
import pandas as pd
import globals
import itertools

#global nb_bd
#global nb_h

def read_meteo(file_path):
    """ Read the weather file and return the features as dataframe
    returns hour, we?, month, G_dh, G_h, Ta, Ts, FF, DD, RH, RR, N
    we indicator (1 of we, 0 if wd) adapted to year 2019 where January 1st was a Tuesday -> modulo of 1,2,3,4,7 are week days
    """
    # Using with ensures that the file is properly closed when you're done
    i = 0
    with open(file_path, 'r') as f:
        we = []
        month = []
        h = []
        G_dh = []
        G_h  = []
        Ta  = []
        Ts  = []
        FF  = []
        DD  = []
        RH  = []
        RR  = []
        N  = []
        for line in f.readlines():
            line = line.split(' ')
            if i>3:
                if (np.mod(int(line[0]), 5) or np.mod(int(line[0]), 6)): boo = 1
                else: boo=0
                we.append(boo)
                month.append(float(line[1]))
                h.append(float(line[2]))
                G_dh.append(float(line[3]))
                G_h.append(float(line[4]))
                Ta.append(float(line[5]))
                Ts.append(float(line[6]))
                FF.append(float(line[7]))
                DD.append(float(line[8]))
                RH.append(float(line[9]))
                RR.append(float(line[10]))
                N.append(float(line[11]))
            i = i+1

    print('Meteo data extraction finished')
    res_meteo = pd.DataFrame({'Hour': h, 'WE?': we, 'Month': month, 'Diffuse hor. irr.': G_dh, 'Normal beam irr.': G_h, 'Air temp': Ta,'Soil temp': Ts, 'Wind speed': FF, 'Wind dir': DD, 'Rel humidity': RH, 'Precipitation': RR, 'Nebulosity': N})
    azimuth, zenith = angle_azimuth()

    return res_meteo, azimuth[:-1], zenith[:-1]


def read_gml(file_path):
    """ Read the gml file and return the features as dataframe
    returns the dataframe with all the attributes and nrj_demand, occ_rate as numpy arrays
    """
    dom = ET.parse(file_path)
    root = dom.getroot()

    constr = []
    co = []
    uVal = []
    for elm in root.findall('./{http://www.opengis.net/gml}featureMember'):
        for cons in elm.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}Construction'):
            constr.append(cons.attrib)
            for u in cons.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}uValue'):
                uVal.append(u.text)
    for a in range(len(constr)):
        for key,value in constr[a].items():
            co.append(value)

    building = [] # building id
    nrj = [] # hourly heating demand over a year []
    gross_vol = [] # Gross Volume of []
    usage_type = [] #residential....
    floor_area = [] # gross floor area from the usage class [m2]
    nb_occ = [] # number of occupants
    heat_conv = [] # convective heat fraction [ratio]
    heat_lat = [] # latent heat fraction [ratio]
    heat_rad = [] # radiant heat fraction [ratio]
    heat_tot = [] # [W] heat dissipation, heat exchange total value
    added_u = []
    inf_rate = []
    wall_area = []
    roof_area = []
    open_area = []
    trans_frac = [] # mean transmittance [fraction] per building
    trans_range = []
    occrate = []
    wall_type = []
    u_value = []
    appliance_rts = []
    appliance_power = []
    posi = []

    # Parcourir le gml et extraire les informations voulues
    for elm in root.findall('./{http://www.opengis.net/citygml/2.0}cityObjectMember'):
        for bui in elm.findall('{http://www.opengis.net/citygml/building/2.0}Building'):
            building.append(bui.attrib)

            for bound in bui.findall('{http://www.opengis.net/citygml/building/2.0}boundedBy'):
                for ground in bound.findall('{http://www.opengis.net/citygml/building/2.0}GroundSurface'):
                    for lod2MS in ground.findall('{http://www.opengis.net/citygml/building/2.0}lod2MultiSurface'):
                        for MS in lod2MS.findall('{http://www.opengis.net/gml}MultiSurface'):
                            for sM in MS.findall('{http://www.opengis.net/gml}surfaceMember'):
                                for P in sM.findall('{http://www.opengis.net/gml}Polygon'):
                                    for e in P.findall('{http://www.opengis.net/gml}exterior'):
                                        for LR in e.findall('{http://www.opengis.net/gml}LinearRing'):
                                            for pos in LR.findall('{http://www.opengis.net/gml}posList'): #7rows, 3cols
                                                posi.append(pos.text)
                                                #print(f'pos text: {pos.text}')

            for d in bui.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}demands'):
                for Ed in d.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}EnergyDemand'):
                    for Ea in Ed.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}energyAmount'):
                        for time in Ea.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}RegularTimeSeries'):
                            for dem in time.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}values'):
                                nrj.append(dem.text)

            for lS in bui.findall('{http://www.sig3d.org/citygml/2.0/building/2.0}lod2Solid'):
                for S in bui.findall('{http://www.sig3d.org/citygml/2.0/building/2.0}Solid'):
                    for e in bui.findall('{http://www.sig3d.org/citygml/2.0/building/2.0}exterior'):
                        for CS in bui.findall('{http://www.sig3d.org/citygml/2.0/building/2.0}CompositeSurface'):
                            for sM in bui.findall('{http://www.sig3d.org/citygml/2.0/building/2.0}surfaceMember'):
                                wall_type.append(sM.text)

            for v in bui.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}volume'):
                for typ in v.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}VolumeType'):
                    for vol in typ.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}value'):
                        gross_vol.append(float(vol.text))

            for u in bui.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}usageZone'):
                for U in u.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}UsageZone'):
                    for uzt in U.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}usageZoneType'):
                        usage_type.append(uzt.text)
                    for f in U.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}floorArea'):
                        for F in f.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}FloorArea'):
                            for Fa in F.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}value'):
                                floor_area.append(float(Fa.text))
                    for o in U.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}occupiedBy'):
                        for O in o.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}Occupants'):
                            for hD in O.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}heatDissipation'):
                                for hE in hD.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}HeatExchangeType'):
                                    for conv in hE.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}convectiveFraction'):
                                        heat_conv.append(conv.text)
                                    for lat in hE.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}latentFraction'):
                                        heat_lat.append(lat.text)
                                    for rad in hE.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}radiantFraction'):
                                        heat_rad.append(rad.text)
                                    for tot in hE.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}totalValue'):
                                        heat_tot.append(tot.text)
                            for nbO in O.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}numberOfOccupants'):
                                nb_occ.append(float(nbO.text))
                            for Orate in O.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}occupancyRate'):
                                for ts in Orate.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}TimeSeriesSchedule'):
                                    for tv in ts.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}timeDependingValues'):
                                        for r in tv.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}RegularTimeSeries'):
                                            for vals in r.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}values'):
                                                occrate.append(vals.text)

                    for h in U.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}has'):
                        for	EA in h.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}ElectricalAppliances'):
                            for	oS in EA.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}operationSchedule'):
                                for	DPS in oS.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}DailyPatternSchedule'):
                                    for	dS in DPS.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}dailySchedule'):
                                        for	SD in dS.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}DailySchedule'):
                                            for	s in SD.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}schedule'):
                                                for	RTS in s.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}RegularTimeSeries'):
                                                    for	val in RTS.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}values'):
                                                        appliance_rts.append(val.text)
                            for	pow in EA.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}electricalPower'):
                                appliance_power.append(pow.text)

            u_sum = 0
            wall = 0
            roof = 0
            opening = 0
            frac_sum = 0
            i = 0
            for t in bui.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}thermalZone'):
                for T in t.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}ThermalZone'):
                    for addU in T.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}additionalThermalBridgeUValue'):
                        added_u.append(float(addU.text))
                    for inf in T.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}infiltrationRate'):
                        inf_rate.append(float(inf.text))
                    for b in T.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}boundedBy'):
                        for tb in b.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}ThermalBoundary'):
                            btype = tb.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}thermalBoundaryType')[0]
                            for barea in tb.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}area'):
                                if btype.text == 'outerWall': wall = wall + float(barea.text)
                                if btype.text == 'roof': roof = roof + float(barea.text)
                                #if btype.text == 'groundSlab': floor = floor + float(barea.text)
                            for op in tb.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}construction'):
                                for key,value in op.items():
                                    nb = value.split('#')[1]
                                u_sum = u_sum + float(uVal[co.index(nb)])*float(barea.text)
                            for cont_open in tb.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}contains'):
                                for op in cont_open.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}ThermalOpening'):
                                    for ar in op.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}area'):
                                        opening = opening +float(ar.text)
                                    for c in op.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}construction'):
                                        for C in c.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}Construction'):
                                            for u in C.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}uValue'):
                                                u_sum = u_sum + float(u.text)*float(ar.text)
                                            for o in C.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}opticalProperties'):
                                                for O in o.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}OpticalProperties'):
                                                    for t in O.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}transmittance'):
                                                        for T in t.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}Transmittance'):
                                                            for frac in T.findall('{http://www.sig3d.org/citygml/2.0/energy/1.0}fraction'):
                                                                    frac_sum = frac_sum + float(frac.text)
                                                                    i = i + 1
                    u_value.append(float(u_sum)/(float(wall)+float(roof)+float(opening)))
                    wall_area.append(float(wall))
                    roof_area.append(float(roof))
                    open_area.append(float(opening))
                    trans_frac.append(float(frac_sum/i))

    print('Gml data extraction finished')

    # Mettre au bon format
    nrj_demand = [] # passer d'un string à un vecteur
    for a in range(len(nrj)):
        nrj_demand.append(nrj[a].split(' '))

    occ_rate = []
    for a in range(len(occrate)):
        occ_rate.append(occrate[a][:-1].split(' '))
    
    appliance = []
    for a in range(len(appliance_rts)):
        appliance.append(appliance_rts[a][:-1].split(' '))

    position1 = []
    for a in range(len(posi)):
        position1.append(posi[a][:-1].split(' '))
    position2 = [None]*len(position1)
    for a in range(len(position1)):
        tmp = []
        for b in range(len(position1[a])):
            position1[a][b] = position1[a][b].replace('\t','')
        for b in range(len(position1[a])):
            if position1[a][b].find('\n')==-1: tmp.append(position1[a][b])
            else: tmp.append(position1[a][b][:-1].split('\n'))
        flat_tmp = []
        for i in tmp:
            if isinstance(i, list):
                for j in i:
                    if j!='': 
                        flat_tmp.append(j)
            elif i!='': 
                flat_tmp.append(i)
        rows = int(len(flat_tmp)/3)
        position2[a] = np.asarray(flat_tmp).reshape((rows, 3))
                
    building_id = [] # passer d'un dictionnaire à un vecteur
    for a in range(len(building)):
        for key,value in building[a].items():
            if value == 'None':
                value = '0'
            building_id.append(float(value))

    position = {"id": building_id, "pos": position2}

    # Create global varibale for the number of buildings and hours
    globals.nb_bd = len(building_id)
    globals.nb_h = np.asarray(nrj_demand).shape[1]

    # Get the heats in [W] - no more fractions
    heat_lat=np.multiply(np.asarray(heat_lat, dtype=np.float32),np.asarray(heat_tot, dtype=np.float32))
    heat_rad=np.multiply(np.asarray(heat_rad, dtype=np.float32),np.asarray(heat_tot, dtype=np.float32))
    heat_conv=np.multiply(np.asarray(heat_conv, dtype=np.float32),np.asarray(heat_tot, dtype=np.float32))

    res_gml = pd.DataFrame({'id': building_id, 'vol': gross_vol, 'usage': usage_type,'floor area': floor_area, 'nb occ': nb_occ, 'tot heat': heat_tot, 'conv heat': heat_conv, 'lat heat': heat_lat, 'rad heat': heat_rad, 'added u': added_u, 'inf rate': inf_rate, 'wall area': wall_area, 'roof area': roof_area, 'open area': open_area, 'transmission': trans_frac, 'occupation rate': occ_rate, 'u value': u_value, 'appliance':appliance, 'appliance_power':appliance_power})
    nrj_demand = np.transpose(np.asarray(nrj_demand, dtype=np.float64), (1, 0))
    return res_gml, nrj_demand, np.asarray(occ_rate, dtype=np.float32), np.asarray(appliance, dtype=np.float32), position

def get_data_3D(gml, meteo, occ_rate, appliance, azimuth, zenith):
    """ Creates one 3D matrix from the gml and meteo dataframes and the occupancy rate
    Also tranforms occupation rate and electrical appliance to get yearly values (and not coefficients)
    3D data : 0 - hours, 1 - attributs, 2 - buildings
    Also takes out the constant attributes: added_u, heat_lat, N here
    """
    # Delete the constant columns
    meteo = meteo.loc[:, (meteo != meteo.iloc[0]).any()]
    gml = gml.loc[:, (gml != gml.iloc[0]).any()]
    #Take out the vector columns and get numpy arrays
    gml_array = gml.drop(['occupation rate', 'appliance', 'appliance_power'], axis=1).to_numpy()
    meteo_array = meteo.to_numpy()

    data = np.zeros((gml_array.shape[0], gml_array.shape[1]+2+meteo.shape[1]+2, meteo_array.shape[0]))
    # repeat all building attributes for each hour
    for i in range(meteo_array.shape[0]):
        data[:, :gml_array.shape[1], i] = gml_array
    # add the occupancy rate wich depends on the building and the hour
    tmp = np.repeat(np.asarray(gml['nb occ'], dtype=np.float32).reshape(globals.nb_bd, 1), globals.nb_h, axis=1)
    data[:, gml_array.shape[1], :] = np.multiply(tmp, occ_rate)
    # Get the electrical appliance values yearly 
    power = np.repeat(np.asarray(gml['appliance_power'], dtype=np.float32).reshape(globals.nb_bd, 1), appliance.shape[1], axis=1)
    appliance = np.repeat(appliance, globals.nb_h/appliance.shape[1], axis=1)
    data[:, gml_array.shape[1]+1, :] = appliance
    
    # repeat all meteo value for each building and add the position
    for i in range(gml_array.shape[0]):
        data[i, gml_array.shape[1]+2:-2, :] = meteo_array.T
        # Add the azimuth and zenith and position angle for each building
        data[i, data.shape[1]-2, :] = azimuth
        data[i, data.shape[1]-1, :] = zenith
    data = np.transpose(data, (2, 1, 0))
    print('3D data matrix finished')
    return data

def get_data_2D(data3D, nrj):
    """ Transform the 3D data to 2D by deleting the hours axis and concatenating the informations
    """
    data2D = np.zeros((data3D.shape[0]*data3D.shape[2], data3D.shape[1]))
    nrj1D = np.zeros((nrj.shape[0]*nrj.shape[1]))
    #for each hour get every info for each building in a row then pass to next hour
    # 3D is hour - attribute - building
    # 2D is building/hour - attribute (with hour id added)
    for i in range(data3D.shape[0]):
        data2D[i*data3D.shape[2]:(i+1)*data3D.shape[2], :data3D.shape[1]+1] = np.transpose(data3D[i, :, :], (1, 0))
        nrj1D[i*data3D.shape[2]:(i+1)*data3D.shape[2]] = nrj[i, :]

    data = pd.DataFrame(data2D, columns=['building_id', 'gross_vol', 'floor_area', 'nb_occ', 'heat_tot', 'heat_conv', 'heat_rad', 'inf_rate', 'wall_area', 'roof_area', 'open_area', 'trans_frac','u_value', 'appliance', 'occ_rate', 'hour', 'we?', 'month', 'G_dh', 'G_h', 'Ta', 'Ts', 'FF', 'DD', 'RH', 'RR', 'azimuth', 'zenith'])
    data['energy']= nrj1D

    print('2D data frame finished')
    return data

########################## DATA TRANSFORMATION ##############################
def transform(data, to_drop=['heat_conv','heat_rad','heat_tot','roof_area','RR'], prev=0, to_prev=['occ_rate','G_dh','G_h','Ta','Ts','FF','DD','RH'], prev_moy=0, prev_E = 1):
    """ Apply the transforms that give the optimal distribution to the data
    """
    data_tr = data.drop(to_drop, axis=1)
    print(f"\n{to_drop} successfully removed")
    if prev == 1 or prev_E == 1: data_tr = get_data_prev(data_tr, to_prev, prev_E, prev)
    if prev_moy == 1: data_tr = get_data_prev_moy(data_tr, to_prev)
    return data_tr

def get_data_prev(data, to_add, prev_E = 1, prev=1):
    """ Adds the previous values of the dataframe to the i-th line
    """
    data_loop = pd.concat([data.iloc[(globals.nb_h*globals.nb_bd-24*globals.nb_h*globals.nb_bd):], data])
    data_new = data.copy()
    times_prev = globals.t_prev
    if prev==1: 
        to_add = to_add
        if prev_E == 1: 
            to_add.append('energy')
    else: 
        if prev_E==1: to_add = ['energy']
    for i, feat in enumerate(to_add):
        for j, time in enumerate(times_prev):
            feat_prev = np.zeros(globals.nb_bd*globals.nb_h)
            for h in range(24, globals.nb_h+24):
                feat_prev[globals.nb_bd*(h-24):globals.nb_bd*(h-24+1)] = np.asarray(data_loop[feat][globals.nb_bd*(h-time):globals.nb_bd*(h-time+1)])
            name = feat + "_prev" + str(time)
            data_new[name] = feat_prev
    print(f'Previous values of {to_add} data  at t-{globals.t_prev} successfully added')
    return data_new

def get_data_prev_moy(data, to_add):
    """ Adds the mean of the previous 24 values of the dataframe to the i-th line
    """
    data_loop = pd.concat([data.iloc[(globals.nb_h*globals.nb_bd-24*globals.nb_h*globals.nb_bd):], data.iloc[:(25*globals.nb_h*globals.nb_bd)]])
    data_new = data.copy()
    for i, feat in enumerate(to_add):
        feat_prev = np.zeros(globals.nb_bd*globals.nb_h)
        for h in range(globals.nb_h):
            feat_sum = np.zeros(globals.nb_bd)
            if h>=24:
                for d in range(1, 25):
                    feat_sum = feat_sum + np.asarray(data[feat][globals.nb_bd*(h-d):globals.nb_bd*(h-d+1)])
            else:
                for d in range(1, 25):
                    feat_sum = feat_sum + np.asarray(data_loop[feat][globals.nb_bd*(h+d):globals.nb_bd*(h+d+1)])
            feat_prev[globals.nb_bd*h:globals.nb_bd*(h+1)] = feat_sum/24
        name = feat + "_prev_moy"
        data_new[name] = feat_prev
    
    print(f'Previous 24 hour mean of {to_add} data successfully added')
    return data_new

################################### TRAIN/TEST SET #######################################
def get_data_train_test(data_tr): 
    """ Returns the dataset separated into train and test
    """
    #Moyenne annuelle pour chaque bat
    mean_nrj=[]
    ID=data_tr['building_id']
    resultantID=[]
    for element in ID:
        if element not in resultantID:
            resultantID.append(element)
    for items in resultantID:
        build=data_tr[data_tr['building_id']==items]
        mean_nrj.append(np.mean(build['energy']))
    df_mean_nrj = pd.DataFrame(mean_nrj,columns =['nrj'])
    df_ID = pd.DataFrame(resultantID).assign(nrj=df_mean_nrj)

    #enlever le min et max
    new_df_ID=df_ID.drop(index=df_ID['nrj'].idxmin())
    new_df_ID=df_ID.drop(index=df_ID['nrj'].idxmax())

    # Create datasets with the quantiles 
    data1=new_df_ID.query('nrj<nrj.quantile([0.25]).values[0]')
    data1.columns=('ID','nrj')
    interm1=new_df_ID.query('nrj>nrj.quantile([0.25]).values[0]')
    data2=interm1[interm1['nrj']<new_df_ID.nrj.quantile([0.5]).values[0]]
    data2.columns=('ID','nrj')
    interm2=new_df_ID.query('nrj>nrj.quantile([0.5]).values[0]')
    data3=interm2[interm2['nrj']<new_df_ID.nrj.quantile([0.75]).values[0]]
    data3.columns=('ID','nrj')
    data4=new_df_ID.query('nrj>nrj.quantile([0.75]).values[0]')
    data4.columns=('ID','nrj')

    # Prendre 30% de chaque classe -> Test set
    data1_30=data1[0:25]
    data2_30=data2[0:25]
    data3_30=data3[0:25]
    data4_30=data4[0:25]
    idx_test = pd.concat([data1_30, data2_30, data3_30, data4_30], axis = 0)
    i = 0
    for idx in idx_test['ID']:
        tmp = data_tr[data_tr['building_id']==idx]
        if i == 0:
            data_test = tmp
        else: 
            data_test = pd.concat([data_test, tmp], axis=0)
        i = i+1
    #data_test = data_tr['building_id']==idx for idx in idx_test]

    # Enlever les batiments tests du train set 
    idx1=[]
    idx2=[]
    idx3=[]
    idx4=[]
    for i in range(data4_30.shape[0]):
        idx1.append(data_tr[data_tr['building_id'] == np.asarray(data1_30['ID'])[i]].index)
        idx2.append(data_tr[data_tr['building_id'] == np.asarray(data2_30['ID'])[i]].index)
        idx3.append(data_tr[data_tr['building_id'] == np.asarray(data3_30['ID'])[i]].index)
        idx4.append(data_tr[data_tr['building_id'] == np.asarray(data4_30['ID'])[i]].index)  
    data_tr1=data_tr
    for idx in idx1:
        data_tr1 = data_tr1.drop(index = list(idx), axis = 0, inplace=False)
    data_tr2=data_tr1
    for idx in idx2:
        data_tr2 = data_tr2.drop(index = list(idx), axis = 0, inplace=False)       
    data_tr3=data_tr2
    for idx in idx3:
        data_tr3 = data_tr3.drop(index = list(idx), axis = 0, inplace=False) 
    data_train=data_tr3
    for idx in idx4:
        data_train = data_train.drop(index = list(idx), axis = 0, inplace=False)

    X_tr = data_train.drop(['energy', 'building_id'], axis=1, inplace=False)
    y_tr = data_train['energy']
    #print(f'X_tr and y_tr in readers: {X_tr}, {y_tr}')
    X_te = data_test.drop(['energy', 'building_id'], axis=1, inplace=False)
    y_te = data_test['energy']
    #print(f'X_te and y_te in readers: {X_te}, {y_te}')
    print('Train and test set successfully finished')
    return X_tr, X_te, y_tr, y_te



