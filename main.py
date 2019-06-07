import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

filename = 'trend12.csv'
dataset = pd.read_csv(filename)

X = dataset['Laikas'].values.reshape(-1,1)
Y = dataset['skaicius'].values.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#To retrieve the intercept:
print("xD")
print(regressor.predict([[20190526]]))
print("2xD")
