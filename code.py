#import the Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression  
#import the data file with the help of pandas .
data = pd.read_csv('provide_ypur_path')
#to understand that it is working just print any statement after loading data.
print("loading the Data")
data
#to know exact size of the data :
data.shape
data.head()
data.tail()
#to get rough information plot scattered graph.
data.plot(kind='scatter',x='Hours',y='Score') 
#to know the more information plot box type graph so that we can compare data easily.
data.plot(kind='box')
plt.show()
#to find correlation coefficients
data.corr()
#change to dataframe variables
hour=pd.DataFrame(data['Hours'])
score=pd.DataFrame(data['Score'])
hour
score
#from the above it comes to know that it should linear dataset.
#Build Linear Regression model
lm=LinearRegression()
model=lm.fit(hour,score)
model.coef_
model.intercept_
model.score(hour,score)
new_hour=np.array([9.25])
new_hour=new_hour.reshape(-1,1)
score_predict=model.predict(new_hour)
score_predict
data.plot(kind='scatter',x='Hours',y='Score')
plt.plot(hour,model.predict(hour),color='green',linewidth='1.5')
plt.scatter(new_hour,score_predict,color='black')
plt.show()


