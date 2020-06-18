# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 00:03:40 2020

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Linear_Regression\\Salary_Data.csv")
df.columns
df.head()
plt.hist(df.Salary)
plt.boxplot(df.Salary,0,'rs',0)
plt.plot(df.Salary,df.YearsExperience,'bo');plt.xlabel("Salary");plt.ylabel("Years of Experience")

#To find the correlation

df.Salary.corr(df.YearsExperience)

np.corrcoef(df.Salary,df.YearsExperience)

#Model building

import statsmodels.formula.api as smf
model=smf.ols("Salary~YearsExperience",data=df).fit()
model
pred=model.predict(df)
pred
model.params
model.summary()
#Data Vizualization

plt.scatter(x=df['YearsExperience'],y=df['Salary'],color='red');plt.plot(df['YearsExperience'],pred,color='black');plt.xlabel("YearsExperience");plt.ylabel("Salary")

pred.corr(df.Salary)
#Tranforming variables for accuracy

model1=smf.ols("Salary~np.log(YearsExperience)",data=df).fit()
model1
model1.summary()
pred1=model1.predict(df)
pred1
pred1.corr(df.Salary)
#Data vizualization:

plt.scatter(x=df['YearsExperience'],y=df['Salary'],color='red');plt.plot(df['YearsExperience'],pred1,color='black');plt.xlabel("YearsExperience");plt.ylabel("Salary")

#Exponential Transformation
model2=smf.ols("np.log(Salary)~YearsExperience",data=df).fit()
model2
model2.summary()
pred2=model2.predict(df)
pred2
pred2.corr(df.Salary)
#Data vizualization:

plt.scatter(x=df['YearsExperience'],y=df['Salary'],color='red');plt.plot(df['YearsExperience'],pred1,color='black');plt.xlabel("YearsExperience");plt.ylabel("Salary")


#Quadgratic Transformation
df['YearsExperience_Sq']=df.YearsExperience*df.YearsExperience
model3=smf.ols("Salary~YearsExperience_Sq",data=df).fit()
model3
model3.summary()
pred3=model3.predict(df)
pred3
pred3.corr(df.Salary)
#Data vizualization:

plt.scatter(x=df['YearsExperience'],y=df['Salary'],color='red');plt.plot(df['YearsExperience'],pred3,color='black');plt.xlabel("YearsExperience");plt.ylabel("Salary")
