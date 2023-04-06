#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from env import user, password, host
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

from acquire import df


# In[2]:


from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score


# In[3]:


df.head()


# In[4]:


#Checking to see if the import worked, looks good


# In[5]:


type(df)


# In[6]:


#Checking the data type


# In[7]:


df_drop = df.drop(['year_built','fips','taxamount'],axis =1)


# In[8]:


#Cutting unnecessary columns


# In[9]:


df = df_drop.astype('int64')


# In[10]:


#Changing the type


# In[11]:


df.size


# In[12]:


df.shape


# In[13]:


#Checking it's size


# In[14]:


df.isnull().any()


# In[15]:


#Checking for any null data


# In[16]:


df.describe().T


# In[17]:


#Reviewing the stats


# In[18]:


df = df[df['tax_value'] != 0]


# In[19]:


#Removing anything that has a null value 


# In[20]:


df.head()


# In[21]:


df.shape


# In[22]:


df['bathrooms'].max()

