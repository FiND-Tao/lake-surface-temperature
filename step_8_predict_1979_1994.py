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
from tqdm import tqdm
model_path="/data/taoliu/taoliufile/great_lake/new_data_derived/model_data/full_data_model/fine_tuned/GLS/GLS_files_model.h5"
model=load_model(model_path)
data=glob.glob('/data/taoliu/taoliufile/great_lake/new_data_derived/point_allyear_temporalize/GLS_79_94/*.sav')
output='/data/taoliu/taoliufile/great_lake/new_data_derived/point_allyear_prediction/GLS_79_94/'
#for file in data:
def process(file):
    y_pred=[]
    dict=pickle.load(open(file,'rb'))
    X=dict['X']
    y_hat=np.squeeze(model.predict(X))
    y_pred.extend(y_hat)
    name=os.path.join(output,os.path.basename(file))
    pickle.dump(y_pred,open(name, 'wb'))
if __name__ == '__main__':
    #p=Pool(20)
    for file in tqdm(data):
        process(file)