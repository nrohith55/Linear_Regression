# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 23:18:18 2020

@author: Rohith
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# reading a csv file using pandas library
calories_consumed=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\calories_consumed.csv")

calories_consumed=calories_consumed.rename(columns={'Weight gained (grams)':"Weight",'Calories Consumed':"Calories"})
calories_consumed
plt.hist(calories_consumed.Weight)
plt.boxplot(calories_consumed.Weight,0,"rs",0)
plt.plot(calories_consumed.Calories,calories_consumed.Weight,"bo");plt.xlabel("Calories");plt.ylabel("Weight")

#To find the correlation
calories_consumed.Weight.corr(calories_consumed.Calories)

np.corrcoef(calories_consumed.Weight,calories_consumed.Calories)


np.corrcoef(calories_consumed.Weight,calories_consumed.Calories)


#To apply linear regression we have to import statsmodels.formula.asi as smf


import statsmodels.formula.api as smf
model=smf.ols("Weight~Calories",data=calories_consumed).fit()
model.params #To ge the model parameters,intercepts

model.summary()#To get the R-Squared value
model.conf_int(0.05)#95%confoidence interval

#To find the predicted value

pred=model.predict(calories_consumed)
pred

#Data Vizua;lization

import matplotlib.pylab as plt

plt.scatter(x=calories_consumed['Calories'],y=calories_consumed['Weight'],color='red');plt.plot(calories_consumed['Calories'],pred,color='black');plt.xlabel("Calories");plt.ylabel("Weight")

pred.corr(calories_consumed.Weight)

#Transforming variables for accuracy

model1=smf.ols("Weight~np.log(Calories)",data=calories_consumed).fit()
model1
model.params
model1.summary()

pred1=model1.predict(calories_consumed)
pred1
#Data Vizualization
import matplotlib.pylab as plt
plt.scatter(x=calories_consumed['Calories'],y=calories_consumed['Weight'],color='red');plt.plot(calories_consumed['Calories'],pred1,color='black');plt.xlabel("Calories");plt.ylabel("Weight")


#Exponential Transformation

model2=smf.ols("np.log(Weight)~Calories",data=calories_consumed).fit()
model2.summary()
pred2=model2.predict(calories_consumed)
pred2
#Data Vizualization

import matplotlib.pylab as plt

plt.scatter(x=calories_consumed["Calories"],y=calories_consumed["Weight"],color='red');plt.plot(calories_consumed["Calories"],pred2,color='black');plt.xlabel("Calories");plt.ylabel("Weight")


#Quadratic transformation

calories_consumed["Calories_Sq"]=calories_consumed.Calories*calories_consumed.Calories
model3=smf.ols('Weight~Calories_Sq',data=calories_consumed).fit()
model3.summary()
pred3=model3.predict(calories_consumed)
pred3
import matplotlib.pylab as plt
#Data Vizulatiztion

plt.scatter(x=calories_consumed["Calories"],y=calories_consumed["Weight"],color='red');plt.plot(calories_consumed['Calories'],pred3,color='black');plt.xlabel("Calories");plt.ylabel("Weight")
