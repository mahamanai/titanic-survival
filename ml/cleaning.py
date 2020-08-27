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

# Check missings
print(dataset.isna().sum()) # Take care of Age and Cabin

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

# Feature engineering 2: Family
dataset['FamilySize'] = dataset['SibSp'] + dataset ['Parch'] + 1
dataset['Family'] = dataset['FamilySize'] > 1
dataset['Family'] = dataset['Family'].astype(int)

# Feature engineering 3: Port
dataset = pd.concat([dataset, pd.get_dummies(dataset['Embarked'])], axis=1)

# Feature engineering 3: Pclass
dataset = pd.concat([dataset, pd.get_dummies(dataset['Pclass'])], axis=1)

# Feature engineering 4: Ticket

# Feature engineering : Age
print(dataset.Age.isna().sum())


# Export cleaned dataset
#dataset.to_csv (r'data/cleaned/titanic.csv', index = False, header=True)
