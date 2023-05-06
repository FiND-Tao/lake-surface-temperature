from __future__ import print_function
import tensorflow as tf    
tf.compat.v1.disable_v2_behavior() # <-- HERE !
import numpy as np
import pandas as pd
from scipy.io import loadmat,savemat
import glob
import os
from inspect import signature
import matplotlib.pyplot as plt
import sklearn.preprocessing
import pickle
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,RepeatVector,TimeDistributed,LSTM
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool

files=glob.glob('/data/taoliu/taoliufile/great_lake/new_data/GLE/data_cfsr*.mat')
files.sort()
files=files[16:]
datamat=loadmat(files[0])['dd']['sw'][0,0]
pNum,_=datamat.shape
output_folder='/data/taoliu/taoliufile/great_lake/new_data_derived/point_allyear/GLE_79_94/'
timesteps = 5
n_features=7
def temporalize(X,  lookback):
    output_X = []
    for i in range(len(X)-lookback+1):
        t = []
        for j in range(0,lookback):
            # Gather past records upto the lookback period
            t.append(X[[(i+j)], :])
        output_X.append(t)
    return output_X

WTcorrectionPath = '/data/taoliu/taoliufile/great_lake/data/GLE_model_data'
file_scalarX=os.path.join(WTcorrectionPath,'scalerX.sav')
file_scalarY=os.path.join(WTcorrectionPath,'scalerY.sav')
scalerX=pickle.load(open(file_scalarX,'rb'))
scalerY=pickle.load(open(file_scalarY,'rb'))
def process(p):
    point_allyear_file=os.path.join(output_folder,'point_{}.sav'.format(p))
    file = open(point_allyear_file,'rb')
    point_allyear_data = pickle.load(file)
    X_std=scalerX.transform(point_allyear_data[:,:7])
    X_temporalize_std=temporalize(X_std,timesteps)
    X_temporalize_std=np.array(X_temporalize_std)
    X_temporalize_std=X_temporalize_std.reshape(X_temporalize_std.shape[0],timesteps,n_features)
    data_item={'X':X_temporalize_std}
    output_folder_dataitem='/data/taoliu/taoliufile/great_lake/new_data_derived/point_allyear_temporalize/GLE_79_94/'
    data_item_name=os.path.join(output_folder_dataitem,'point_{}.sav'.format(p))
    pickle.dump(data_item,open(data_item_name, 'wb'))

if __name__ == '__main__':
    p=Pool()
    p.map(process, range(pNum))