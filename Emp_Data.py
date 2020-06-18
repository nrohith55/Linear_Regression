# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 23:27:07 2020

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Linear_Regression\\emp_data.csv")
#Emp_data -> Build a prediction model for Churn_out_rate
#y=Churn_out_rate ; x=Salary_hike
plt.hist(df.Churn_out_rate)
plt.boxplot(df.Churn_out_rate)
plt.plot(df.Salary_hike,df.Churn_out_rate,"bo");plt.xlabel("Salary_hike");plt.ylabel("Churn_out_rate")

#Model_building:
import statsmodels.formula.api as smf
model=smf.ols("Churn_out_rate~Salary_hike",data=df).fit()
model.summary()
pred=model.predict(df)
pred
#Data Vizualization
plt.scatter(x=df.Salary_hike,y=df.Churn_out_rate,color='red');plt.plot(df.Salary_hike,pred,color='black');plt.xlabel("Salary_hike");plt.ylabel("Churn_out_rate")

pred.corr(df.Churn_out_rate)

#Applying transformations on variables to get accuracy

model1=smf.ols("Churn_out_rate~np.log(Salary_hike)",data=df).fit()
model1.summary()
pred1=model1.predict(df)
pred1
#Data Vizualization
plt.scatter(x=df.Salary_hike,y=df.Churn_out_rate,color='red');plt.plot(df.Salary_hike,pred1,color='black');plt.xlabel("Salary_hike");plt.ylabel("Churn_out_rate")

pred1.corr(df.Churn_out_rate)


#Exponential transformation

model2=smf.ols("np.log(Churn_out_rate)~Salary_hike",data=df).fit()
model2.summary()
pred2=model2.predict(df)
pred2
#Data Vizualization
plt.scatter(x=df.Salary_hike,y=df.Churn_out_rate,color='red');plt.plot(df.Salary_hike,pred2,color='black');plt.xlabel("Salary_hike");plt.ylabel("Churn_out_rate")

pred2.corr(df.Churn_out_rate)

#Quadratic transformation
df["Salary_hike_Sq"]=df.Salary_hike*df.Salary_hike
model3=smf.ols("Churn_out_rate~Salary_hike_Sq",data=df).fit()
model3.summary()
pred3=model3.predict(df)
pred3
#Data Vizualization
plt.scatter(x=df.Salary_hike,y=df.Churn_out_rate,color='red');plt.plot(df.Salary_hike,pred3,color='black');plt.xlabel("Salary_hike");plt.ylabel("Churn_out_rate")

pred3.corr(df.Churn_out_rate)





