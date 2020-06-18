# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 02:13:56 2020

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Linear_Regression\\delivery_time.csv")
df=df.rename(columns={"Delivery Time":"deliverytime","Sorting Time":"sortingtime"})

#here y=deliverytime x=sortingtime

plt.hist(df.deliverytime)
plt.boxplot(df.deliverytime,0,"rs",0)
plt.plot(df.sortingtime,df.deliverytime,"bo");plt.xlabel("sortingtime");plt.ylabel("deliverytime")

#To find correlation
df.sortingtime.corr(df.deliverytime)
np.corrcoef(df.sortingtime,df.deliverytime)

#Model building

import statsmodels.formula.api as sfm
model =  sfm.ols("deliverytime~sortingtime",data=df).fit()
model.summary()
pred=model.predict(df)
pred
pred.corr(df.deliverytime)
#Data vizualization:

plt.scatter(x=df['sortingtime'],y=df['deliverytime'],color='red');plt.plot(df['sortingtime'],pred,color='black');plt.xlabel("sortingtime");plt.ylabel("deliverytime")


#Transformating variables for accuracy

model1 =  sfm.ols("deliverytime~np.log(sortingtime)",data=df).fit()
model1.summary()
pred1=model1.predict(df)
pred1
pred1.corr(df.deliverytime)
#Data vizualization:

plt.scatter(x=df['sortingtime'],y=df['deliverytime'],color='red');plt.plot(df['sortingtime'],pred1,color='black');plt.xlabel("sortingtime");plt.ylabel("deliverytime")

#Exponential transformation method
model2 =  sfm.ols("np.log(deliverytime)~sortingtime",data=df).fit()
model2.summary()
pred2=model2.predict(df)
pred2
pred2.corr(df.deliverytime)
#Data vizualization:

plt.scatter(x=df['sortingtime'],y=df['deliverytime'],color='red');plt.plot(df['sortingtime'],pred2,color='black');plt.xlabel("sortingtime");plt.ylabel("deliverytime")


#Quadgratic transformation method
df['sortingtime_Sq']=df.sortingtime*df.sortingtime
model3 =  sfm.ols("deliverytime~sortingtime_Sq",data=df).fit()
model3.summary()
pred3=model3.predict(df)
pred3
pred3.corr(df.deliverytime)
#Data vizualization:

plt.scatter(x=df['sortingtime'],y=df['deliverytime'],color='red');plt.plot(df['sortingtime'],pred3,color='black');plt.xlabel("sortingtime");plt.ylabel("deliverytime")
