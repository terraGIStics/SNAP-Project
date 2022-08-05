#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import missingno as msno

from sklearn.decomposition import PCA
import autoreload


# In[55]:


#New Mexico
nm07 = pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/clean_nm07.csv')
nm17 = pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/clean_nm17.csv')

#Nebraska
ne07 = pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/clean_ne07.csv')
ne17 = pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/clean_ne17.csv')


# In[56]:


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

def plot_simple_features(column,img_name,description):
    """Plots features on a countplot, used for columns with binary values and for EDA datasets."""
    plt.figure(figsize = (16,10))
    plt.suptitle(description, fontsize=20)
    idx = 221
    while idx<225:
        for key,value in {'nm07':nm07_orig,'nm17':nm17_orig,'ne07':ne07_orig,'ne17':ne17_orig}.items():
            mean = value[column].mean()
            ax = plt.subplot(idx)
            plt.title(f'$\it{key.upper()}$')
            sns.countplot(value[column],palette="husl")
            ax.axhline(mean,linewidth=1,color='r')
            ax.set_xlabel('')
            #ax.set_xticklabels([0,1])
            idx +=1
    plt.savefig("C:/Users/Casey/Desktop/SNAP/Images/Ind_Features/" + str(img_name) + ".png")
    

def plot_features(column,img_name,description):
    """Plots features on a countplot, used for columns with binary values."""
    plt.figure(figsize = (16,10))
    plt.suptitle(description, fontsize=20)
    idx = 221
    while idx<225:
        for key,value in {'nm07':nm07,'nm17':nm17,'ne07':ne07,'ne17':ne17}.items():
            ax = plt.subplot(idx)
            plt.title(f'$\it{key.upper()}$')
            sns.countplot(value[column])
            ax.axhline(y=value[value[column]==1][column].size,linewidth=1,color='r')
            ax.set_xlabel('')
            ax.set_xticklabels([0,1])
            idx +=1
    plt.savefig("C:/Users/Casey/Desktop/SNAP/Images/Ind_Features/" + str(img_name) + ".png")

def plot_features_hist(column,img_name,description):
    """Plots features on a histogram, best for currency columns"""
    plt.figure(figsize = (16,10))
    plt.suptitle(description, fontsize=20)
    idx = 221
    while idx<225:
        for key,value in {'nm07':nm07,'nm17':nm17,'ne07':ne07,'ne17':ne17}.items():
            ax = plt.subplot(idx)
            plt.title(f'$\it{key.upper()}$')
            plt.hist(value[column],bins=20,range=(1,value[column].max()))
            ax.set_xlabel(f"Number of zero's:{value[value[column]==0][column].count()}")
            ax.xaxis.set_label_coords(0.15, 1.05)
            idx +=1
    plt.savefig("C:/Users/Casey/Desktop/SNAP/Images/Ind_Features/" + str(img_name) + ".png")
    
def final(fullname, dataset):
    """Returns a sliced dataset of columns found in corr_features list."""
    fullname_df = pd.DataFrame()
    for feature in fullname:
        try:
            fullname_df[feature] = dataset[feature]
        except:
            continue
    fullname_df['CAT_ELIG']=dataset['CAT_ELIG']
    return fullname_df


# In[57]:


###################################
         # 2007 Datasets
###################################

unit07_demo= pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/07_DataDict/UNIT_Demo.csv')
unit07_assets=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/07_DataDict/UNIT_Assets.csv')
unit07_exded=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/07_DataDict/UNIT_ExDed.csv')
unit07_inc=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/07_DataDict/UNIT_Inc.csv')
per07_char=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/07_DataDict/PERS_Char.csv')
per07_inc=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/07_DataDict/PERS_Inc.csv')


# In[58]:


###################################
         # 2017 Datasets
###################################

unit17_demo= pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/17_DataDict/UNIT_Demo.csv')
unit17_assets=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/17_DataDict/UNIT_Assets.csv')
unit17_exded=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/17_DataDict/UNIT_ExDed.csv')
unit17_inc=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/17_DataDict/UNIT_Inc.csv')
per17_char=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/17_DataDict/PERS_Char.csv')
per17_inc=pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/17_DataDict/PERS_Inc.csv')


# In[59]:


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


# In[60]:


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


# In[61]:


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


# In[62]:


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


# In[63]:


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


# In[64]:


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


# In[65]:


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


# In[66]:


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


# In[ ]:


##### Observations #####

# 1. Is the family receiving TANF benefits.
# 2. Is the head of household receiving TANF benefits?
# 3. Is the head of household receiving social security (SSI):
# 4. Are other members in the household receiving social security?

# This tells us that the biggest impact on vulnerable communities within New Mexico Nebraska are due to the head of 
# households personal characteristics. Out of that, we see the biggest impact on SNAP eligibility is receiving benefits 
# from other assistance programs, especially TANF (Temporary Assistance for Needy Families). 
# What is interesting is that is a slightly bigger factor for New Mexico residents over Nebraska residents. 
# And more households/head of households were receiving TANF in 2007. 
# So maybe that accounts for wider eligibility of applicants in 2017, because less people in the SNAP program was 
# receiving TANF income.


# In[22]:


plot_features_hist('FSTANF','unit_tanf',"Countable TANF Income for Head of Household")


# In[23]:


plot_features_hist('TANF1','pers_tanf1',"Countable TANF Income for Head of Household")


# In[24]:


plot_features_hist('FSUNEARN','unit_unearn',"Total Unearned Income per Unit")


# In[72]:


plot_features('WRK_POOR','wrk_pr',"Working Poor")


# In[73]:


plt.figure(figsize = (12,7))
sns.countplot(ne07['VEHICLEA'])
plt.xticks(np.arange(8),labels=[1,2,3,4,5,6,7,8])
plt.xlabel("Vehicle Category")
plt.title("NE07 Vehicle A Category:\n (1)None | (2-5)Exempt | (6-8)NOT Exempt")
plt.savefig("C:/Users/Casey/Desktop/SNAP/Images/Ind_Features/vehA.png");


# In[74]:


##### Correlated Features #####


# In[75]:


set1 = list(pd.DataFrame(corr_df(unit07_demo, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5)).T.columns)
set2 = list(pd.DataFrame(corr_df(unit07_assets, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5)).T.columns)
set3 = list(pd.DataFrame(corr_df(unit07_exded, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5)).T.columns)
set4 = list(pd.DataFrame(corr_df(unit07_inc, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5)).T.columns)
set5 = list(pd.DataFrame(corr_df(per07_char, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5)).T.columns)
set6 = list(pd.DataFrame(corr_df(per07_inc, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5)).T.columns)

set7 = list(pd.DataFrame(corr_df(unit07_demo, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5)).T.columns)
set8 = list(pd.DataFrame(corr_df(unit07_assets, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5)).T.columns)
set9 = list(pd.DataFrame(corr_df(unit07_exded, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5)).T.columns)
set10 = list(pd.DataFrame(corr_df(unit07_inc, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5)).T.columns)
set11 = list(pd.DataFrame(corr_df(per07_char, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5)).T.columns)
set12 = list(pd.DataFrame(corr_df(per07_inc, nm07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5)).T.columns)

corr_features = set1 + set2 + set3 + set4 + set5 + set6 + set7 + set8 + set9 + set10 + set11


# In[76]:


set1 = list(pd.DataFrame(corr_df(unit07_demo, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5)).T.columns)
set2 = list(pd.DataFrame(corr_df(unit07_assets, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5)).T.columns)
set3 = list(pd.DataFrame(corr_df(unit07_exded, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5)).T.columns)
set4 = list(pd.DataFrame(corr_df(unit07_inc, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5)).T.columns)
set5 = list(pd.DataFrame(corr_df(per07_char, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5)).T.columns)
set6 = list(pd.DataFrame(corr_df(per07_inc, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=False).head(5)).T.columns)

set7 = list(pd.DataFrame(corr_df(unit07_demo, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5)).T.columns)
set8 = list(pd.DataFrame(corr_df(unit07_assets, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5)).T.columns)
set9 = list(pd.DataFrame(corr_df(unit07_exded, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5)).T.columns)
set10 = list(pd.DataFrame(corr_df(unit07_inc, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5)).T.columns)
set11 = list(pd.DataFrame(corr_df(per07_char, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5)).T.columns)
set12 = list(pd.DataFrame(corr_df(per07_inc, ne07).corr()[['CAT_ELIG']].sort_values(by = 'CAT_ELIG',ascending=True).head(5)).T.columns)

corr_features = set1 + set2 + set3 + set4 + set5 + set6 + set7 + set8 + set9 + set10 + set11
corr_features=set(corr_features)
corr_features.remove('CAT_ELIG')


# In[77]:


print(f'There are {len(corr_features)} features in corr_features:\n')
corr_features


# In[78]:


##### Final Dataset #####


# In[79]:


corr_features=list(corr_features)
pd.DataFrame(corr_features,columns=['Table 1']).to_csv("C:/Users/Casey/Desktop/SNAP/Data/corr_features.csv",index=None)


# In[80]:


df1 = final(corr_features,nm07)
df2 = final(corr_features,nm17)
df3 = final(corr_features,ne07)
df4 = final(corr_features,ne17)


# In[81]:


df = pd.concat([df1,df2,df3,df4])


# In[82]:


df.info()


# In[83]:


df['VEHICLEA']=df['VEHICLEA'].fillna(1)


# In[84]:


df.isnull().sum()


# In[85]:


df = df.fillna(0)


# In[86]:


df.isnull().sum().sum()


# In[87]:


df.to_csv('C:/Users/Casey/Desktop/SNAP/Data/final.csv',index=None)


# In[ ]:




