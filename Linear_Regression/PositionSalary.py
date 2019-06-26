import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

plt.scatter(x,y,color='red')
#%matplotlib auto # by pressing control+enter and then delete

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

linRegressor = LinearRegression()
linRegressor.fit(x,y)


polyFeatures = PolynomialFeatures(degree=2)

newX = polyFeatures.fit_transform(x)

polyFeatures.fit(newX,y)

linRegressorNew = LinearRegression()
linRegressorNew.fit(newX,y)


plt.scatter(x,y,color='red')
plt.plot(x,linRegressorNew.predict(newX),'b')

y_pred = linRegressorNew.predict(newX)


plt.scatter(x,y,color='blue')
plt.plot(x,linRegressor.predict(x),color='red')
plt.show()

z=[3.5]
newz=np.array(z).reshape(1,-1)
y_newpred = linRegressor.predict(newz)
newz = polyFeatures.fit_transform(newz)

a=linRegressorNew.score(newX,y)

i=2
while a < 0.99:
    polyFeatures = PolynomialFeatures(degree=i)
    newX = polyFeatures.fit_transform(x)
    linRegressorNew = LinearRegression()
    linRegressorNew.fit(newX,y)
    a=linRegressorNew.score(newX,y)
    plt.scatter(x,y,color='blue')
    plt.plot(x,linRegressorNew.predict(newX),color='red')
    i+=1
plt.show()    

    

    