import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

filename = 'canada_per_capita_income.csv'
dataset = pd.read_csv(filename)





X = dataset['income'].values.reshape(-1,1)
Y = dataset['year'].values.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#To retrieve the intercept:
print("Answer")

predict = regressor.predict([[32738.2629]])
print(predict)
print("Data")
print(X)
plt.figure()
plt.subplot(2,1,1)
plt.plot(X, Y)
plt.title('Original data')


plt.subplot(2,1,2)
plt.plot(X, X)
plt.title('Predicted')
plt.show()