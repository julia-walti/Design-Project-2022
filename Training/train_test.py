import os
from jinja2 import ModuleLoader
import numpy as np
import globals 
import helpers 
import readers 
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import joblib

if __name__ == "__main__": 
    globals.initialize() 

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

    #################################### MODELS ###############################################
    # Linear regression
    lr = linear_model.LinearRegression()

    # Random Forest
    rf = RandomForestRegressor(n_estimators = 50,random_state = 42)

    # Multilayer percetron
    #['activation' : ['identity', 'logistic', 'tanh', 'relu'],
    mlp = MLPRegressor(max_iter=200,
                        tol = 10000,    
                        n_iter_no_change = 3,
                        hidden_layer_sizes=(10,),
                        random_state=42,
                        verbose=True,
                        shuffle = True,
                        learning_rate_init=1e-2,
                        learning_rate = 'adaptive')
    pipe_mlp = Pipeline([('scaler', MinMaxScaler()), ('mlp', mlp)])

    # TPOT Results
    tpot1 = DecisionTreeRegressor(max_depth=10, min_samples_leaf=19, min_samples_split=14)
    tpot2 = Pipeline([('scaler', StandardScaler()),('DTR', DecisionTreeRegressor(max_depth=10, min_samples_leaf=1, min_samples_split=4))])
    tpot3 = RandomForestRegressor(bootstrap=True, max_features=0.7500000000000001, min_samples_leaf=11,  min_samples_split=9, n_estimators=15)
    tpot4 = DecisionTreeRegressor(max_depth=6, min_samples_leaf=17, min_samples_split=5)

    ################################ TRAINING ######################################################
    # /!\ CHOICES /!\
    globals.prev = 0
    globals.prev_e = 0
    to_drop_Ta = ['azimuth', 'zenith', 'gross_vol', 'nb_occ', 'heat_tot', 'heat_conv', 'heat_rad', 'inf_rate', 'wall_area', 'roof_area', 'open_area', 'trans_frac', 'u_value','appliance', 'occ_rate', 'hour', 'we?', 'month', 'G_dh', 'G_h', 'Ts', 'FF', 'DD','RH', 'RR']
    to_drop = []
    to_prev = ['occ_rate','G_dh','G_h','Ta','Ts','FF','DD','RH', 'appliance']
    globals.t_prev = [] #[1, 2, 3, 4, 5, 8, 10]
    model = tpot3 # Choose between : lr, rf, mlp, tpots
    globals.n_hours = 24

    # Transform data
    data_tr = readers.transform(data2D, to_drop = to_drop, prev = globals.prev, to_prev = to_prev, prev_E = globals.prev_e, prev_moy= globals.prev_mean)
    # Get the vectors for the training
    X_tr, X_te, y_tr, y_te = readers.get_data_train_test(data_tr)
    print(f"\nStart of the training of the {model}\nwith prev={globals.prev}, prev_mean = {globals.prev_mean}, prev_e = {globals.prev_e}")

    # Training with real energy values
    model.fit(X_tr.values, y_tr.values)
    # Save the model 
    joblib.dump(model, 'A_Training/model.joblib')
    print('Model saved')

    ################################# TESTING ######################################################
    globals.area_te = np.asarray(X_te['floor_area'])
    if to_drop == to_drop_Ta: X_te = X_te.drop(['floor_area'], axis = 1)
    day_test = {'nb':[80-1, 172-1, 264-1, 355-1], 'name': ['March 21st', 'June 21st', 'September 21st', 'December 21st']}
    abs_e = np.zeros((len(day_test['nb']), globals.n_hours, globals.nb_bd_te))
    pred = np.zeros((len(day_test['nb']), globals.n_hours, globals.nb_bd_te))
    Y_te = np.zeros((globals.nb_h, globals.nb_bd_te))
    Y_box = np.zeros((len(day_test['nb']), globals.n_hours, globals.nb_bd_te))
    A_te = np.zeros((globals.nb_h, globals.nb_bd_te))
    for b in range(globals.nb_bd_te):
        Y_te[:, b] = y_te[b*globals.nb_h:(b+1)*globals.nb_h] 
        A_te[:, b] = globals.area_te[b*globals.nb_h:(b+1)*globals.nb_h] 
    for i, d in enumerate(day_test['nb']): 
        print(f"\nStart test with use of computed energy for day {d}")
        pred[i, :, :], abs_e[i, :, :] = helpers.Test_lagged_n_hours(X_te, Y_te, model, d, n_hours=globals.n_hours)
        for b in range(globals.nb_bd_te):
            Y_box[i, :, b] = Y_te[d*24:d*24+globals.n_hours, b]

