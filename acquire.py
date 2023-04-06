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

def acquire_zillow():
    if os.path.exists('zillow_2017.csv'):
        return pd.read_csv('zillow_2017.csv', index_col=0)
    else:
        ''' Acquire data from Zillow using env imports and rename columns'''

        url = f"mysql+pymysql://{user}:{password}@{host}/zillow"

        query = """
        SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
        FROM properties_2017
        LEFT JOIN propertylandusetype USING(propertylandusetypeid)
        WHERE propertylandusedesc IN ("Single Family Residential",                       
                                      "Inferred Single Family Residential")"""

        df = pd.read_sql(query, url)


        df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                                  'bathroomcnt':'bathrooms', 
                                  'calculatedfinishedsquarefeet':'area',
                                  'taxvaluedollarcnt':'tax_value', 
                                  'yearbuilt':'year_built',})
        return df


# In[2]:


#Importing all the things needed as well as the necessary columns for the data


# In[3]:


zillow = acquire_zillow().dropna


# In[4]:


#Replacing all null data with 0 to look less awkward


# In[5]:


zillow_new = zillow().reset_index()


# In[6]:


zillow_new


# In[7]:


df = zillow_new.astype('int64')


# In[8]:


df


# In[9]:


df_sample = df.head(1000)


# In[21]:


def bathroomtax(df): 
    plt.figure(figsize=(16,8))
    sns.barplot(data=df_sample,x='bathrooms',y='tax_value')
    plt.show()


# In[22]:


def bedroomstax(df):
    plt.figure(figsize=(16,8))
    sns.barplot(data=df_sample,x='bedrooms',y='tax_value')
    plt.show()


# In[24]:


def squaretax(df):
    plt.figure(figsize=(16,8))
    sns.scatterplot(data=df_sample,x='square_feet',y='tax_value')
    plt.show()


# In[25]:


def squarerooms(df):
    plt.figure(figsize=(16,8))
    sns.scatterplot(data=df_sample,x='square_feet',y='bedrooms')
    plt.show()


# In[26]:


def bedbathsq(df):
    plt.figure(figsize=(16,8))
    sns.lineplot(data=df_sample,x='bedrooms',y='tax_value')
    sns.lineplot(data=df_sample,x='bathrooms',y='tax_value')
    plt.legend(labels=["Bed rooms","Bed Divengence","Bath rooms","Bath Divengence"])
    plt.show()


# In[27]:


def pairsquare(df):
    sns.pairplot(df_sample)
    plt.show()


# In[28]:


def taxguess(df):
    plt.figure(figsize=(16,8))
    plt.hist(y_train.tax_value,label="Total Value")
    plt.hist(y_train.tv_pred_mean, bins=1,label="Predicted Tax value")
    plt.xlabel("Final value")
    plt.ylabel("Total")
    plt.legend()
    plt.show()


# In[29]:


def predict(df):
    plt.figure(figsize=(16,8))

    plt.plot(y_validate.tax_value, y_validate.tv_pred_mean, alpha=.5, color="gray", label='_nolegend_')
    plt.annotate("Baseline", (16, 9.5))

    plt.plot(y_validate.tax_value, y_validate.tax_value, alpha=.5, color="blue", label='_nolegend_')
    plt.annotate("Ideal Line", (.5, 3.5), rotation=15.5)

    plt.scatter(y_validate.tax_value, y_validate.tv_pred_mean, 
            alpha=.5, color="red", s=100, label="Linear Regression")

    plt.scatter(y_validate.tax_value, y_validate.tv_pred_median, 
            alpha=.5, color="purple", s=100, label="Lasso Lars")

    plt.scatter(y_validate.tax_value, y_validate.tax_value, 
            alpha=.5, color="yellow", s=100, label="Tweedie Regressor")

    plt.scatter(y_validate.tax_value, y_validate.tax_value, 
            alpha=.5, color="green", s=100, label="Polynomial")
    plt.legend()
    plt.xlabel("Actual Total")
    plt.ylabel("Predicted Total")
    plt.title("Predictions most extreme?")
    plt.show()


# In[31]:


def predictionofsome(df):
    plt.figure(figsize=(16,8))

    plt.hist(y_validate.tax_value, color='blue', alpha=.5, label="Actual Total")
    plt.hist(y_validate.tax_value, color='red', alpha=.5, label="Linear Regression")
    plt.hist(y_validate.tax_value, color='purple', alpha=.5, label="Lasso Lars")

    plt.xlabel("Total")
    plt.ylabel("Tax value")
    plt.title("Comparing the Distribution")
    plt.legend()
    plt.show()


# In[ ]:




