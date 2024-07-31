# lake-surface-temperature
Step 1 Data Preparation: Generate one file for every single point, and this file contains 'sw','lw','lh','sh','t2','u10','v10', ‘lst’ on its corresponding point from year 1995 to 2020.

Data can be downloaded from https://drive.google.com/drive/folders/1u8K2iFyDQCG8bVojHQwWH9dz8yk9ITkp?usp=sharing

Step 2 sample generation: generate data samples from step 1 output. Each sample contains the features for five consecutive years and the y value for the last year.

Step 3: split the samples into training and testing sets.

Step 4: train the model for each lake.

Step 5: generate the evaluation plot.

Step 6: data preparation for year 1979 to 1994

Step 7: generate the samples for year 1979 to 1994.

Step 8: predict surface temperature for year 1979 to 1994
