#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#from __future__ import print_function
import tensorflow as tf    
tf.compat.v1.disable_v2_behavior() # <-- HERE !
import numpy as np
import pandas as pd
from scipy.io import loadmat,savemat
import glob
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
import sklearn.metrics as sm
model_path="/data/taoliu/taoliufile/great_lake/new_data_derived/model_data/train_test_model_scratch/GLH/GLH_files_model.h5"
model=load_model(model_path)
data=pd.read_csv('/data/taoliu/taoliufile/great_lake/new_data_derived/GLH_files_test.csv')
data=data.squeeze().values.tolist()

file=data[700]
dict=pickle.load(open(file,'rb'))
X=dict['X']
Y=dict['Y']    
y_true=Y
name=os.path.basename(file)[:-4]
WTcorrectionPath = '/data/taoliu/taoliufile/great_lake/data/GLE_model_data'
file_scalarX=os.path.join(WTcorrectionPath,'scalerX.sav')
file_scalarY=os.path.join(WTcorrectionPath,'scalerY.sav')
scalerX=pickle.load(open(file_scalarX,'rb'))
scalerY=pickle.load(open(file_scalarY,'rb'))

y_true=scalerY.inverse_transform(np.expand_dims(np.array(y_true),1))

file_pre=r'/data/taoliu/taoliufile/great_lake/new_data_derived/point_allyear_prediction_individual_lake_train_scratch_79_20_csv/GLH/{}.csv'.format(name)
data_pre=pd.read_csv(file_pre)
y_pred=data_pre.loc[data_pre['year']>=1995,'temperature'][4:]

plt.plot(y_pred,y_true,'ro')

y_test=y_true
y_test_pred=y_pred
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2)) 
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
# %%
