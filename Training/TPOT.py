import os
import numpy as np
import globals 
import helpers 
import readers 
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from tpot import TPOTRegressor
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from tpot.config import regressor_config_dict

import multiprocessing

if __name__ == "__main__": 
    globals.initialize() 

    # /!\ CHOICES /!\
    globals.prev = 0
    globals.prev_mean = 0
    globals.prev_e = 1

    regressor_config_dict['tpot.builtins.FeatureSetSelector'] = {'n_estimators': [10, 20, 30],
            'min_samples_split': [2, 9], 
            'min_samples_leaf': [1, 11],
            'max_features': ['auto', 0.7500000000000001]
            }

    tpot = TPOTRegressor(generations=1, population_size=20, #<-- change according to prior knowledge of the problem and dimensions
                        offspring_size=None, mutation_rate=0.9,
                        crossover_rate=0.1, scoring='neg_mean_squared_error', cv=5,
                        subsample=1.0, n_jobs=1, #<-- njobs=-1 uses all the available cores in the CPU
                        max_time_mins=None, max_eval_time_mins=5,
                        random_state=42, warm_start=True,
                        memory=None, use_dask=False,
                        periodic_checkpoint_folder='/Archives', #<-- folder of partially optimized pipelines, usefull for recovery
                        early_stop=None, verbosity=3,
                        disable_update_check=False
                        )#regressor_config_dict, 
                        #template='Selector-Transformer-RandomForestRegressor')

    # You can also change prev, prev_mean, prev_E at line 18
    # base case : to_drop = ['heat_conv', 'heat_rad', 'heat_tot', 'RR']
    to_drop = ['roof_area', 'trans_frac', 'u_value', 'occ_rate', 'we?', 'month', 'hour', 'RR', 'azimuth', 'zenith']
    to_prev = ['occ_rate','G_dh','G_h','Ta','Ts','FF','DD','RH', 'appliance']
    globals.t_prev = [1, 2, 3, 4, 5, 8, 10]

    gml_name = 'satom_calibrated_energy.gml'
    gml_path = os.path.join('A_Training/data', gml_name)
    meteo_name = 'Aigle_MeteoSchweiz_2019.cli'
    meteo_path = os.path.join('A_Training/data', meteo_name)

    print('')
    gml, nrj_dem, occ_rate, appliance_rts, pos = readers.read_gml(gml_path)
    meteo, azimuth, zenith = readers.read_meteo(meteo_path)
    globals.e_mean = [np.mean(nrj_dem, axis=1), meteo['Hour']]
    data3D = readers.get_data_3D(gml, meteo, occ_rate, appliance_rts, azimuth, zenith)
    data2D = readers.get_data_2D(data3D, nrj_dem)
     
    # Transform data
    data_tr = readers.transform(data2D, to_drop = to_drop, prev = globals.prev, to_prev = to_prev, prev_E = globals.prev_e, prev_moy= globals.prev_mean)
    # Get the vectors for the training
    X_tr, X_te, y_tr, y_te = readers.get_data_train_test(data_tr)

    multiprocessing.set_start_method('forkserver')
    print(f"\nStart of the training of the Tpot\nwith prev={globals.prev}, prev_mean = {globals.prev_mean}, prev_e = {globals.prev_e}\n")                            
    tpot.fit(np.asarray(X_tr),np.asarray(y_tr)) #<--your numpy features and target time-series vector
    print(f'Score:{ tpot.score(np.asarray(X_te), np.asarray(y_te))}')
    tpot.export('tpot_Model.py')#<--optimal pipeline is finally saved here


