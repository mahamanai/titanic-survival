import pandas as pd

def read_cleaned_dat():
    dataset = pd.read_csv('./data/cleaned/titanic.csv')
    X = dataset.iloc[:, dataset.columns != 'Survived'].values
    y = dataset.iloc[:, 1].values
    return X