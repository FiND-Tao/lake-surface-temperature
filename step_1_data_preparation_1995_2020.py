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

files=glob.glob('/data/taoliu/taoliufile/great_lake/new_data/GLH/data_cfsr*.mat')
files.sort()
files=files[16:] # select the files from year 1995
datamat=loadmat(files[0])['dd']['sw'][0,0]
pNum,_=datamat.shape
var_names = ['sw','lw','lh','sh','t2','u10','v10']
year=files[0][-8:-4]
y_file='/data/taoliu/taoliufile/great_lake/new_data/GLH/data_GLSEA_glh_{}.mat'.format(year)
ymat=loadmat(y_file)['dd']['lst'][0,0]
output_folder='/data/taoliu/taoliufile/great_lake/new_data_derived/point_allyear/GLH/'
def process(p):
#    print(p)
    p_data=[]
    for file in files:
        year_p_tmp=[]# data for year # and point #
        for var in var_names:
            datamat=loadmat(file)['dd'][var][0,0] #6106x365 we have 6106 points
            t1=datamat[p,:]
            year_p_tmp.append(t1)
        year=file[-8:-4]
        y_file='/data/taoliu/taoliufile/great_lake/new_data/GLH/data_GLSEA_glh_{}.mat'.format(year)
        ymat=loadmat(y_file)['dd']['lst'][0,0]  
        t2=ymat[p,:]  
        year_p_tmp.append(t2)
        a=np.array(year_p_tmp)
        b=np.transpose(a)
        p_data.append(np.transpose(np.array(year_p_tmp))) #365x8
    p_data_array=np.concatenate(p_data,axis=0) 
    pickle.dump(p_data_array,open(output_folder+'/point_{}.sav'.format(p), 'wb'))  

if __name__ == '__main__':
    p=Pool()
    p.map(process, range(pNum))

