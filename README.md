# stock_price_prediction
import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key ='R96RNAsSU_ysrdtJieMQ'
df=quandl.get("WIKI/AAPL")
df=df[['Adj. Close']]
df.tail()

df['Adj. Close'].plot(figsize= (20,6), color='b')
plt.legend(loc='upper right')
plt.show()

forecast = 35
df['prediction']=df['Adj. Close'].shift(-forecast)
x=np.array(df.drop(['prediction'],1))
x=preprocessing.scale(x)

x_forecast=x[-forecast:]
x=x[:-forecast]

y=np.array(df['prediction'])
y=y[:-forecast]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

clf = LinearRegression()
clf.fit(x_train,y_train)

confidence = clf.score(x_test,y_test)
forecast_predicted=clf.predict(x_forecast)
print(forecast_predicted)

dates = pd.date_range(start='2018-03-28' , end='2018-05-01')
plt.plot(dates,forecast_predicted,color='r')
df['Adj. Close'].plot(color='g')
plt.xlim(xmin=datetime.date(2013,3,30))
plt.show()
