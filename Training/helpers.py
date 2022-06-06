import numpy as np
import globals 
import datetime as dt
import pandas as pd
import math 
import pvlib #conda install -c pvlib pvlib
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
import joblib
import random

################################### ZENITH ANGLE #########################################
def angle_azimuth():
    """Computes the zenith and azimuth angles
    """
    h = dt.datetime(2019,1,1,0,0,0)
    h_vec=[h]
    for i in range(8760):
        h_vec.append(h_vec[i]+dt.timedelta(hours=1))
    lat=(46+(14+(19.4/60))/60)*math.pi/180
    L=(6+(57+(33.1/60))/60)*math.pi/180
    h_frame=pd.DatetimeIndex(h_vec)
    z=pvlib.solarposition.get_solarposition(h_frame, lat, L, altitude=None, pressure=None, method='nrel_numpy', temperature=12)
    azimuth=z['azimuth']
    zenith=z['zenith']
    return np.asarray(azimuth), np.asarray(zenith)

################################## TESTING ###########################################
def Test_lagged_n_hours(X_te, Y_te, model, day, n_hours):
    """ Test the model over a certain number of hours for certain days 
    """
    pred_lag = np.zeros((n_hours, globals.nb_bd_te))
    abs_e = np.zeros((n_hours, globals.nb_bd_te))

    for b in range(globals.nb_bd_te):
        X_te_lag = X_te.iloc[b*globals.nb_h+day*24:b*globals.nb_h+day*24+n_hours].copy() #len is n_hours, already for the right building and hours
        y = Y_te[day*24:day*24+n_hours, b].copy()
        if globals.prev_e == 1:
            for j, time in enumerate(globals.t_prev):
                e_loop = np.zeros(n_hours)
                e_loop[:time] = Y_te[day*24-time:day*24, b] 
                name = "energy_prev" + str(time)
                X_te_lag[name] = e_loop
    
        # Lagged testing loop
        for h in range(n_hours):
            pred = model.predict(np.asarray(X_te_lag.iloc[h].values).reshape(1, len(X_te_lag.iloc[h].values)))
            if pred < 0: pred=0
            pred_lag[h, b] = pred
            abs_e[h, b] = np.abs(pred-y[h])
            if globals.prev_e == 1: 
                for j, time in enumerate(globals.t_prev):
                    name = "energy_prev" + str(time)
                    if h+time < len(X_te_lag): 
                        X_te_lag[name].iloc[h+time] = pred

    return pred_lag, abs_e



