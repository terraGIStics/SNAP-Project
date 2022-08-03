#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import missingno as msno

from sklearn.decomposition import PCA
import autoreload


# In[2]:


#New Mexico
nm07 = pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/clean_nm07.csv')
nm17 = pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/clean_nm17.csv')

#Nebraska
ne07 = pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/clean_ne07.csv')
ne17 = pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/clean_ne17.csv')


# In[3]:


###### Correlation Dataframe Codes - All Codes in this Cell ######

def corr_df(fullname, dataset):
    """Creates a dataframe on .corr(), used for unit categories."""
    fullname_df = pd.DataFrame()
    for feature in fullname['Table 1']:
        try:
            fullname_df[feature] = dataset[feature]
        except:
            continue
    fullname_df['CAT_ELIG']=dataset['CAT_ELIG'].astype("float64")
    return fullname_df

def corr_numcol(fullname,dataset):
    """Creates a dataframe on .corr(), used for personal characteristics categories."""
    fullname_df=pd.DataFrame()
    for feature in fullname['Table 1']:
        feature = feature[:-1]
        for num in range(1,17):
            combo=str(feature+str(num))
            try:
                fullname_df[combo]=dataset[combo]
            except:
                continue
    fullname_df['CAT_ELIG']=dataset['CAT_ELIG'].astype('float64')
    return fullname_df


# In[4]:


###################################
         # 2007 Datasets
###################################

unit07_demo= pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/07_DataDict/UNIT_Demo.csv')
unit07_assets=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/07_DataDict/UNIT_Assets.csv')
unit07_exded=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/07_DataDict/UNIT_ExDed.csv')
unit07_inc=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/07_DataDict/UNIT_Inc.csv')
per07_char=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/07_DataDict/PERS_Char.csv')
per07_inc=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/07_DataDict/PERS_Inc.csv')


# In[5]:


###################################
         # 2017 Datasets
###################################

unit17_demo= pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/17_DataDict/UNIT_Demo.csv')
unit17_assets=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/17_DataDict/UNIT_Assets.csv')
unit17_exded=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/17_DataDict/UNIT_ExDed.csv')
unit17_inc=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/17_DataDict/UNIT_Inc.csv')
per17_char=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/17_DataDict/PERS_Char.csv')
per17_inc=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/17_DataDict/PERS_Inc.csv')


# In[6]:


plt.figure(figsize= (25,10))
plt.suptitle("2007 New Mexico Correlations (Descending)",fontsize=15)
plt.subplot(2,3,1)
sns.heatmap(corr_df(unit07_demo, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True,xticklabels=[])
plt.title("Unit Demographics")
plt.yticks(rotation=0)
plt.subplot(2,3,2)
sns.heatmap(corr_df(unit07_assets, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Assets")
plt.yticks(rotation=0)
plt.subplot(2,3,3)
sns.heatmap(corr_df(unit07_exded, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Expenses/Deductions")
plt.yticks(rotation=0)
plt.subplot(2,3,4)
sns.heatmap(corr_df(unit07_inc, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Income")
plt.yticks(rotation=0)
plt.subplot(2,3,5)
sns.heatmap(corr_numcol(per07_char, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Personal Charcteristics")
plt.yticks(rotation=0)
plt.subplot(2,3,6)
sns.heatmap(corr_numcol(per07_inc, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Personal Income")
plt.yticks(rotation=0)
plt.savefig('C:/Users/Casey/Desktop/SNAP/Images/Correlations/corrD_nm07.png');


# In[7]:


plt.figure(figsize= (25,10))
plt.suptitle("2007 New Mexico Correlations (Ascending)",fontsize=15)
plt.subplot(2,3,1)
sns.heatmap(corr_df(unit07_demo, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True,xticklabels=[])
plt.title("Unit Demographics")
plt.yticks(rotation=0)
plt.subplot(2,3,2)
sns.heatmap(corr_df(unit07_assets, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Assets")
plt.yticks(rotation=0)
plt.subplot(2,3,3)
sns.heatmap(corr_df(unit07_exded, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Expenses/Deductions")
plt.yticks(rotation=0)
plt.subplot(2,3,4)
sns.heatmap(corr_df(unit07_inc, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Income")
plt.yticks(rotation=0)
plt.subplot(2,3,5)
sns.heatmap(corr_numcol(per07_char, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Personal Charcteristics")
plt.yticks(rotation=0)
plt.subplot(2,3,6)
sns.heatmap(corr_numcol(per07_inc, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Personal Income")
plt.yticks(rotation=0)
plt.savefig('C:/Users/Casey/Desktop/SNAP/Images/Correlations/corrA_nm07.png');


# In[8]:


plt.figure(figsize= (25,10))
plt.suptitle("2007 Nebraska Correlations(Descending)",fontsize=15)
plt.subplot(2,3,1)
sns.heatmap(corr_df(unit07_demo, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True,xticklabels=[])
plt.title("Unit Demographics")
plt.yticks(rotation=0)
plt.subplot(2,3,2)
sns.heatmap(corr_df(unit07_assets, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Assets")
plt.yticks(rotation=0)
plt.subplot(2,3,3)
sns.heatmap(corr_df(unit07_exded, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Expenses/Deductions")
plt.yticks(rotation=0)
plt.subplot(2,3,4)
sns.heatmap(corr_df(unit07_inc, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Income")
plt.yticks(rotation=0)
plt.subplot(2,3,5)
sns.heatmap(corr_numcol(per07_char, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Personal Charcteristics")
plt.yticks(rotation=0)
plt.subplot(2,3,6)
sns.heatmap(corr_numcol(per07_inc, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Personal Income")
plt.yticks(rotation=0)
plt.savefig('C:/Users/Casey/Desktop/SNAP/Images/Correlations/corrD_ne07.png');


# In[9]:


plt.figure(figsize= (25,10))
plt.suptitle("2007 Nebraska Correlations(Ascending)",fontsize=15)
plt.subplot(2,3,1)
sns.heatmap(corr_df(unit07_demo, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True,xticklabels=[])
plt.title("Unit Demographics")
plt.yticks(rotation=0)
plt.subplot(2,3,2)
sns.heatmap(corr_df(unit07_assets, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Assets")
plt.yticks(rotation=0)
plt.subplot(2,3,3)
sns.heatmap(corr_df(unit07_exded, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Expenses/Deductions")
plt.yticks(rotation=0)
plt.subplot(2,3,4)
sns.heatmap(corr_df(unit07_inc, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Income")
plt.yticks(rotation=0)
plt.subplot(2,3,5)
sns.heatmap(corr_numcol(per07_char, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Personal Charcteristics")
plt.yticks(rotation=0)
plt.subplot(2,3,6)
sns.heatmap(corr_numcol(per07_inc, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Personal Income")
plt.yticks(rotation=0)
plt.savefig('C:/Users/Casey/Desktop/SNAP/Images/Correlations/corrA_ne07.png');


# In[10]:


plt.figure(figsize= (25,10))
plt.suptitle("2017 New Mexico Correlations (Descending)",fontsize=15)
plt.subplot(2,3,1)
sns.heatmap(corr_df(unit17_demo, nm17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True,xticklabels=[])
plt.title("Unit Demographics")
plt.yticks(rotation=0)
plt.subplot(2,3,2)
sns.heatmap(corr_df(unit17_assets, nm17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Assets")
plt.yticks(rotation=0)
plt.subplot(2,3,3)
sns.heatmap(corr_df(unit17_exded, nm17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Expenses/Deductions")
plt.yticks(rotation=0)
plt.subplot(2,3,4)
sns.heatmap(corr_df(unit17_inc, nm17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Income")
plt.yticks(rotation=0)
plt.subplot(2,3,5)
sns.heatmap(corr_numcol(per17_char, nm17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Personal Charcteristics")
plt.yticks(rotation=0)
plt.subplot(2,3,6)
sns.heatmap(corr_numcol(per17_inc, nm17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Personal Income")
plt.yticks(rotation=0)
plt.savefig('C:/Users/Casey/Desktop/SNAP/Images/Correlations/corrD_nm17.png');


# In[11]:


plt.figure(figsize= (25,10))
plt.suptitle("2017 New Mexico Correlations (Ascending)",fontsize=15)
plt.subplot(2,3,1)
sns.heatmap(corr_df(unit17_demo, nm17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True,xticklabels=[])
plt.title("Unit Demographics")
plt.yticks(rotation=0)
plt.subplot(2,3,2)
sns.heatmap(corr_df(unit17_assets, nm17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Assets")
plt.yticks(rotation=0)
plt.subplot(2,3,3)
sns.heatmap(corr_df(unit17_exded, nm17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Expenses/Deductions")
plt.yticks(rotation=0)
plt.subplot(2,3,4)
sns.heatmap(corr_df(unit17_inc, nm17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Income")
plt.yticks(rotation=0)
plt.subplot(2,3,5)
sns.heatmap(corr_numcol(per17_char, nm17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Personal Charcteristics")
plt.yticks(rotation=0)
plt.subplot(2,3,6)
sns.heatmap(corr_numcol(per17_inc, nm17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Personal Income")
plt.yticks(rotation=0)
plt.savefig('C:/Users/Casey/Desktop/SNAP/Images/Correlations/corrA_nm17.png');


# In[12]:


plt.figure(figsize= (25,10))
plt.suptitle("2017 Nebraska Correlations(Descending)",fontsize=15)
plt.subplot(2,3,1)
sns.heatmap(corr_df(unit17_demo, ne17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True,xticklabels=[])
plt.title("Unit Demographics")
plt.yticks(rotation=0)
plt.subplot(2,3,2)
sns.heatmap(corr_df(unit17_assets, ne17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Assets")
plt.yticks(rotation=0)
plt.subplot(2,3,3)
sns.heatmap(corr_df(unit17_exded, ne17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Expenses/Deductions")
plt.yticks(rotation=0)
plt.subplot(2,3,4)
sns.heatmap(corr_df(unit17_inc, ne17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Income")
plt.yticks(rotation=0)
plt.subplot(2,3,5)
sns.heatmap(corr_numcol(per17_char, ne17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Personal Charcteristics")
plt.yticks(rotation=0)
plt.subplot(2,3,6)
sns.heatmap(corr_numcol(per17_inc, ne17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Personal Income")
plt.yticks(rotation=0)
plt.savefig('C:/Users/Casey/Desktop/SNAP/Images/Correlations/corrD_ne17.png');


# In[13]:


plt.figure(figsize= (25,10))
plt.suptitle("2017 Nebraska Correlations(Ascending)",fontsize=15)
plt.subplot(2,3,1)
sns.heatmap(corr_df(unit17_demo, ne17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True,xticklabels=[])
plt.title("Unit Demographics")
plt.yticks(rotation=0)
plt.subplot(2,3,2)
sns.heatmap(corr_df(unit17_assets, ne17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Assets")
plt.yticks(rotation=0)
plt.subplot(2,3,3)
sns.heatmap(corr_df(unit17_exded, ne17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Expenses/Deductions")
plt.yticks(rotation=0)
plt.subplot(2,3,4)
sns.heatmap(corr_df(unit17_inc, ne17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Unit Income")
plt.yticks(rotation=0)
plt.subplot(2,3,5)
sns.heatmap(corr_numcol(per17_char, ne17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Personal Charcteristics")
plt.yticks(rotation=0)
plt.subplot(2,3,6)
sns.heatmap(corr_numcol(per17_inc, ne17).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5),vmin=-1,cmap='viridis',annot=True)
plt.title("Personal Income")
plt.yticks(rotation=0)
plt.savefig('C:/Users/Casey/Desktop/SNAP/Images/Correlations/corrA_ne17.png');

