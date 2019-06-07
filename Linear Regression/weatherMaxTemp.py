import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

filename = 'Weather.csv'
dataset = pd.read_csv(filename)





X = dataset['MinTemp'].values.reshape(-1,1)
Y = dataset['MaxTemp'].values.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#To retrieve the intercept:
print("Answer")
predict = regressor.predict([[21.66666667]])
print(predict)
plt.figure()
plt.subplot(2,1,1)
plt.scatter(X, Y, color='blue', marker='+')
plt.title('Original data')


plt.show()

