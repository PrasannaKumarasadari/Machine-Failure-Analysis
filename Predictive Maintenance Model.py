#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import the Required Libries


 In[1]:import sys
if sys.version_info[0:2] != (2, 6):
raise Exception('Requires python 2.6')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.linear_model import Ridge,Lasso,RidgeCV,LassoCV,ElasticNet,ElasticNetCV, LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport


# In[ ]:


#Loading the Data


# In[2]:


df = pd.read_csv("/Users/LENOVO/Documents/data science/ai4i2020.csv")


# In[3]:


df


# In[ ]:


#Creating Detailed Analysis Using Pandas Profiling


# In[4]:


pf = ProfileReport(df)


# In[6]:


Report = pf.to_widgets()
Report


# In[ ]:


#Saving the Analysis Data to HTML


# In[11]:


pf.to_file('Detailed Analysis Report.html')


# In[ ]:


#Dropping the Not reuried Columns


# In[14]:


df.drop(columns = 'UDI', inplace = True)


# In[15]:


df.drop(columns = 'Product ID', inplace = True)


# In[16]:


df


# In[ ]:


#Converting Type Column to Numeric Notation


# In[23]:


df['Type'] = df['Type'].map({'H':'3','M':'2','L':'1'})


# In[24]:


df


# In[ ]:


#Selecting label and features


# In[25]:


y = df['Air temperature [K]']


# In[27]:


y


# In[57]:


x = df.drop(columns = 'Air temperature [K]')


# In[58]:


x = x.drop(columns = 'Machine failure')


# In[84]:


x


# In[ ]:


#Standaizing the data


# In[30]:


scaler = StandardScaler()


# In[70]:


stand = scaler.fit_transform(x)


# In[80]:


df2 = pd.DataFrame(stand)


# In[81]:


df2


# In[62]:


ProfileReport(df2)


# In[ ]:


#Checking Multi-Colinearity Using Infulenece Factor


# In[53]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vf = pd.DataFrame()


# In[54]:


vf['vif'] = [variance_inflation_factor(stand,i) for i in range(stand.shape[1])]


# In[55]:


vf['feature'] = x.columns


# In[56]:


vf


# In[48]:


#At the initial analysis ViF for column machine Failure is Greater than 10, Column is dropped


# In[ ]:


#Checking Multi-Co-linearity Using Influence Factor


# In[67]:


vf1 = pd.DataFrame()


# In[71]:


vf1['vif'] = [variance_inflation_factor(stand,i) for i in range(stand.shape[1])]


# In[72]:


vf1['feature'] = x.columns


# In[73]:


vf1


# In[ ]:


#Model Training And Validation


# In[142]:


x_train,x_test,y_train,y_test = train_test_split(stand,y,test_size = 0.30,random_state = 2000)


# In[77]:


linear = LinearRegression()


# In[143]:


linear.fit(x_train,y_train)


# In[144]:


pickle.dump(linear,open('predictive_main.pickle','wb'))


# In[129]:


linear.predict(ty)


# In[148]:


ty = scaler.transform([[1,308.6,1551,42.8,1,1,1,1,0,0]])


# In[89]:


ty


# In[145]:


linear.score(x_test,y_test)


# In[147]:


P_model = pickle.load(open('predictive_main.pickle','rb'))


# In[149]:


P_model.predict(ty)


# In[ ]:


#Regularization of Model


# In[150]:


# Let's create a function to create adjusted R-Squared
def adj_r2(x,y):
    r2 = linear.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2


# In[151]:


adj_r2(x_test,y_test)


# In[ ]:


#Lasso test


# In[152]:


lassocv = LassoCV(alphas=None,cv= 50 , max_iter=200000, normalize=True)
lassocv.fit(x_train,y_train)


# In[153]:


lasso = Lasso(alpha=lassocv.alpha_)
lasso.fit(x_train,y_train)


# In[154]:


lasso.score(x_test,y_test)


# In[ ]:


#Rridge Regression


# In[155]:


ridgecv = RidgeCV(alphas=np.random.uniform(0,10,50),cv = 10 , normalize=True)
ridgecv.fit(x_train,y_train)


# In[156]:


ridge_lr = Ridge(alpha=ridgecv.alpha_)
ridge_lr.fit(x_train,y_train)


# In[157]:


ridge_lr.score(x_test,y_test)


# In[ ]:


# Using Elastic CV


# In[158]:


elastic= ElasticNetCV(alphas=None, cv = 10 )
elastic.fit(x_train,y_train)


# In[159]:


elastic_lr = ElasticNet(alpha=elastic.alpha_ , l1_ratio=elastic.l1_ratio_)


# In[160]:


elastic_lr.fit(x_train,y_train)


# In[161]:


elastic_lr.score(x_test,y_test)


# In[ ]:




