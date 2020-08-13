# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dat_train = pd.read_csv('../../data/raw/train.csv')
dat_test = pd.read_csv('../../data/raw/test.csv')

# Putting train and test data together
dat_test.insert(1, "Survived", np.nan)
dat_train['DataTrain'] = 1
dat_test['DataTrain'] = 0
print(dat_train)
print(dat_test)
dataset = pd.concat([dat_train, dat_test])
print(dataset)

# Export cleaned dataset
dataset.to_csv (r'../../data/cleaned/titanic.csv', index = False, header=True)
