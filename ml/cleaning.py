# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dat_train = pd.read_csv('data/raw/train.csv')
dat_test = pd.read_csv('data/raw/test.csv')

# Putting train and test data together
dat_test.insert(1, "Survived", np.nan)
dat_train['DataTrain'] = 1
dat_test['DataTrain'] = 0
print(dat_train)
print(dat_test)
dataset = pd.concat([dat_train, dat_test])

# Feature engineering 1: Honorifics/title indicators
dataset['Titles'] = dataset['Name'].str.extract(r'\, ((\w+\s?)+)\.')[0]
print(dataset)
print(dataset['Titles'].unique())
print(dataset.Titles.value_counts())
dataset = pd.concat([dataset, pd.get_dummies(dataset['Titles'])], axis=1)
print(dataset)
dataset['Army'] = dataset['Col'] + dataset['Major'] + dataset['Capt']
dataset['UnmarriedWoman'] = dataset['Miss'] + dataset['Mlle']
dataset['FormalTitles'] = dataset['the Countess'] + dataset['Lady'] + dataset['Sir']
dataset = dataset.drop(['Mr', 'Miss', 'Mrs', 'Col', 'Major', 'Mlle', 'Ms', 'the Countess','Lady', 'Mme', 'Dona', 'Capt', 'Sir', 'Don', 'Jonkheer'], axis = 1) 
print(dataset)

# Export cleaned dataset
dataset.to_csv (r'data/cleaned/titanic.csv', index = False, header=True)
