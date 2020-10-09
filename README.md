# Prediction-using-superwised-ML
#Predict the Percentage of marks of an student based on the number of study hours
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression  
data = pd.read_csv('C:/Users/ashish/Documents/score_hours.csv')
print("loading the Data")
data
data.shape
data.plot(kind='scatter',x='Hours',y='Score')  
data.plot(kind='box')
plt.show()
#to find correlation coefficients
data.corr()
#change to dataframe variables
hour=pd.DataFrame(data['Hours'])
score=pd.DataFrame(data['Score'])
hour
score
#Build Linear Regression model
lm=LinearRegression()
model=lm.fit(hour,score)
model.coef_
model.intercept_
model.score(hour,score)
import numpy as np
new_hour=np.array([9.25])
new_hour=new_hour.reshape(-1,1)
score_predict=model.predict(new_hour)
score_predict
data.plot(kind='scatter',x='Hours',y='Score')
plt.plot(hour,model.predict(hour),color='green',linewidth='1.5')
plt.scatter(new_hour,score_predict,color='black')
plt.show()
