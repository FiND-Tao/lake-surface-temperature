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
from keras.models import load_model
import matplotlib.pyplot as plt
from multiprocessing import Pool

files=glob.glob('/data/taoliu/taoliufile/great_lake/new_data/GLS/data_cfsr*.mat')
files.sort()
files=files[:16]
datamat=loadmat(files[0])['dd']['sw'][0,0]
pNum,_=datamat.shape
var_names = ['sw','lw','lh','sh','t2','u10','v10']
output_folder='/data/taoliu/taoliufile/great_lake/new_data_derived/point_allyear/GLS_79_94/'
#files=files[:2]
#for p in range(pNum):
def process(p):
#    print(p)
    p_data=[]
    for file in files:
        year_p_tmp=[]# data for year # and point #
        for var in var_names:
            datamat=loadmat(file)['dd'][var][0,0] #6106x365
            t1=datamat[p,:]
            year_p_tmp.append(t1)
        p_data.append(np.transpose(np.array(year_p_tmp)))
    p_data_array=np.concatenate(p_data,axis=0) 
    pickle.dump(p_data_array,open(output_folder+'/point_{}.sav'.format(p), 'wb'))  

if __name__ == '__main__':
    p=Pool(30)
    p.map(process, range(pNum))


