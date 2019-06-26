import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,0:1].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state = 0,test_size=1/3)

from sklearn.linear_model import LinearRegression
sRegressor = LinearRegression()

sRegressor.fit(x_train,y_train)

y_pred = sRegressor.predict(x_test)

plt.scatter(x_train,y_train,color='blue')
plt.plot(x_train,sRegressor.predict(x_train),'r')

uInput = input("Enter the experience")
uList = uInput.split(" ")
nList = []
for i in uList:
    nList.append(float(i))
nData = np.array(nList)
nData = nData.reshape((len(nData),1))
sRegressor.predict(nData)
