import glob
import pandas as pd
import pickle
import numpy as np
import random

random.seed(0)

files=glob.glob('/data/taoliu/taoliufile/great_lake/new_data_derived/point_allyear_temporalize/GLS/*.sav')
random.shuffle(files)
total_files.extend(files)
train_rate=0.8
numOfTrain=int(len(files)*train_rate)
files_train=files[:numOfTrain]
files_test=files[numOfTrain:]
print(len(files))
print(len(files_test))
print(len(files_train))
total_train.extend(files_train)
total_test.extend(files_test)

df=pd.DataFrame({'files':files})
df.to_csv('/data/taoliu/taoliufile/great_lake/new_data_derived/GLS_files.csv',index=False)

df=pd.DataFrame({'files':files_train})
df.to_csv('/data/taoliu/taoliufile/great_lake/new_data_derived/GLS_files_train.csv',index=False)

df=pd.DataFrame({'files':files_test})
df.to_csv('/data/taoliu/taoliufile/great_lake/new_data_derived/GLS_files_test.csv',index=False)
