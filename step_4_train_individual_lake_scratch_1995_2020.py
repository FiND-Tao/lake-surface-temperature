from __future__ import print_function
import tensorflow as tf    
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
class My_Custom_Generator(keras.utils.Sequence) :
      
  def __init__(self, filenames, batch_size) :
    self.filenames = filenames
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_files=self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    
    batchx=[]
    batchy=[]
    for i in batch_files:
        dict=pickle.load(open(i,'rb'))
        X=dict['X']
        Y=dict['Y']
        batchx.append(X)
        batchy.append(Y)

    batchx=np.concatenate(batchx,axis=0)
    batchy=np.concatenate(batchy,axis=0)
    
    return batchx,batchy
n_features=7
timesteps = 5
bS = 2048
itr = 200

def RNN_LSTM_atm(timesteps,n_features):    
  # define model
  model = Sequential()

  model.add(LSTM(32, activation='tanh', kernel_initializer='normal', input_shape=(timesteps,n_features), return_sequences=True))
  model.add(BatchNormalization())
  model.add(Dropout(0.2))

  model.add(LSTM(16, activation='tanh', return_sequences=True))
  model.add(BatchNormalization())
  model.add(Dropout(0.2))

  model.add(LSTM(8, activation='tanh', return_sequences=False))

  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  ##model1.summary()

  return model
for k in ['GLE','GLH','GLM','GLO','GLS']:
  batch_size=14
  model= RNN_LSTM_atm(timesteps,n_features)
  data=pd.read_csv('/data/taoliu/taoliufile/great_lake/new_data_derived/{}_files.csv'.format(k))
  data=data.squeeze().values.tolist()
  my_training_batch_generator = My_Custom_Generator(data, batch_size)

  data_test=pd.read_csv('/data/taoliu/taoliufile/great_lake/new_data_derived/{}_files_test.csv'.format(k))
  data_test=data_test.squeeze().values.tolist()
  my_test_batch_generator = My_Custom_Generator(data_test, batch_size)
  print("number of training samples is {}".format(len(data)))

  history=model.fit(my_training_batch_generator,
                    steps_per_epoch = int(len(data) // batch_size),
                    epochs = 500,
                    verbose = 1)

  model.save('/data/taoliu/taoliufile/great_lake/new_data_derived/model_data/individual_lake_model_scratch/{}/{}_files_model.h5'.format(k,k))
  with open('/data/taoliu/taoliufile/great_lake/new_data_derived/model_data/individual_lake_model_scratch/{}/{}_files_trainHistoryDict'.format(k,k), 'wb') as file_pi:
      pickle.dump(history.history, file_pi)
  print('ok')