#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer


# In[2]:


# 2007 Data = snap07
snap07 = pd.read_spss('C:/Users/Casey/Desktop/SNAP/qcfy2007_spss/qc_pub_fy2007.sav')


# In[3]:


snap07.head(5)


# In[4]:


snap07.shape


# In[5]:


snap07.info()


# In[6]:


snap07.isnull().sum()


# In[7]:


# 2017 Data = snap17
snap17 = pd.read_csv('C:/Users/Casey/Desktop/SNAP/qcfy2017_csv/qc_pub_fy2017.csv')


# In[8]:


snap17.head(5)


# In[9]:


snap17.info()


# In[10]:


snap17.isnull().sum()


# In[11]:


#### Target Variable ####


# In[12]:


# 2007: 1 = Eligible, 2 = Not eligible
snap07['CAT_ELIG'].value_counts()


# In[13]:


#2017: 0 = Not eligible, 1 = Reported eligible, 2 = Recorded eligible
snap17['CAT_ELIG'].value_counts()


# In[14]:


# changing the target variable in both datasets to a dictionary of:
# 0 = Not eligible
# 1 = Eligible
snap07['CAT_ELIG'] = snap07['CAT_ELIG'].replace(2,0)
snap17['CAT_ELIG'] = snap17['CAT_ELIG'].replace(2,1)


# In[15]:


snap07['CAT_ELIG'].value_counts()


# In[16]:


snap17['CAT_ELIG'].value_counts()


# In[17]:


print(f'2017 dataset: {snap17.shape} VS 2007 dataset: {snap07.shape}')


# In[18]:


### Less people nationally applied for SNAP benefits in 2017 as opposed to 2007 (45530 vs 47469). Likely due to stronger
### stonger economic factors such as employment. Note: 45 columns of features were added to the dataset.


# In[19]:


### Time to extract state data from New Mexico and Nebraska ###


# In[20]:


# save New Mexico records (2007)
nm07 = snap07.loc[snap07['STATE'] == 35].astype('float64')
nm07_target = nm07['CAT_ELIG']
nm07.to_csv('C:/Users/Casey/Desktop/SNAP/Data/nm07.csv',index=None)


# In[21]:


# save New Mexico records (2017)
nm17 = snap17.loc[snap17['STATE'] == 35]
nm17_target = nm17['CAT_ELIG']
nm17 = nm17.drop(columns = ['STATENAME'])
nm17 = nm17.astype('float64')
nm17.to_csv('C:/Users/Casey/Desktop/SNAP/Data/nm17.csv',index=None)


# In[23]:


# NM dataframe 2007
df_nm07 = pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/nm07.csv')
df_nm07


# In[25]:


# NM dataframe 2017
df_nm17 = pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/nm17.csv')
df_nm17


# In[26]:


df_nm07['CAT_ELIG'].value_counts()


# In[27]:


df_nm17['CAT_ELIG'].value_counts()


# In[28]:


# save Nebraska records (2007)
ne07 = snap07.loc[snap07['STATE'] == 31].astype('float64')
ne07_target = ne07['CAT_ELIG']
ne07.to_csv('C:/Users/Casey/Desktop/SNAP/Data/ne07.csv',index=None)


# In[29]:


# save Nebraska records (2017)
ne17 = snap17.loc[snap17['STATE'] == 31]
ne17_target = ne17['CAT_ELIG']
ne17 = ne17.drop(columns = ['STATENAME'])
ne17 = ne17.astype('float64')
ne17.to_csv('C:/Users/Casey/Desktop/SNAP/Data/ne17.csv',index=None)


# In[31]:


# NE dataframe 2007
df_ne07 = pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/ne07.csv')
df_ne07


# In[33]:


# NE dataframe 2017
df_ne17 = pd.read_csv('C:/Users/Casey/Desktop/SNAP/Data/ne17.csv')
df_ne17


# In[34]:


df_ne07['CAT_ELIG'].value_counts()


# In[35]:


df_ne17['CAT_ELIG'].value_counts()


# In[36]:


### Null Values ###
# This section is dedicated to reduction of columns by different treatments of null values


# In[37]:


# nm07 removing empty columns
with pd.option_context('display.max_rows', None):
    print(df_nm07.isnull().sum().sort_values(ascending=False)[:96])


# In[38]:


#Remove those columns.
first_cut = df_nm07.isnull().sum().sort_values(ascending=False)[:96]
first_cut_df = pd.DataFrame([first_cut])
first_cut_df.T
fc_list = list(first_cut_df.columns)

#remove them
nm07 = df_nm07.drop(fc_list,axis=1)
nm07


# In[39]:


# nm17 removing empty columns
with pd.option_context('display.max_rows', None):
    print(df_nm17.isnull().sum().sort_values(ascending=False)[:165])


# In[40]:


#Remove those columns.
first_cut = df_nm17.isnull().sum().sort_values(ascending=False)[:165]
first_cut_df = pd.DataFrame([first_cut])
first_cut_df.T
fc_list = list(first_cut_df.columns)

#remove them
nm17 = df_nm17.drop(fc_list,axis=1)
nm17


# In[41]:


# ne07 removing empty columns
with pd.option_context('display.max_rows', None):
    print(df_ne07.isnull().sum().sort_values(ascending=False)[:144])


# In[42]:


#Remove those columns.
first_cut = df_ne07.isnull().sum().sort_values(ascending=False)[:144]
first_cut_df = pd.DataFrame([first_cut])
first_cut_df.T
fc_list = list(first_cut_df.columns)

#remove them
ne07 = df_ne07.drop(fc_list,axis=1)
ne07


# In[43]:


# ne17 removing empty columns
with pd.option_context('display.max_rows', None):
    print(df_ne17.isnull().sum().sort_values(ascending=False)[:91])


# In[44]:


#Remove those columns.
first_cut = df_ne17.isnull().sum().sort_values(ascending=False)[:91]
first_cut_df = pd.DataFrame([first_cut])
first_cut_df.T
fc_list = list(first_cut_df.columns)

#remove them
ne17 = df_ne17.drop(fc_list,axis=1)
ne17


# In[45]:


### Partial Missing: High Nullity ###
# use a 50% cutoff of missing rows in a column to ensure the imputation method is more accurate.
# this method will run congruently for all four dataframes


# In[46]:


dict_df = {'nm07':nm07,'nm17':nm17,'ne07':ne07,'ne17':ne17}


# In[47]:


for key, value in dict_df.items():
    print(f'50% mark for high nullitary columns:')
    print(f'{key}: {round(value.shape[0]/2)}')


# In[48]:


all_df_att = pd.DataFrame(dict_df.keys(), columns = ['name'])
all_df_att['rows'] = [value.shape[0] for key,value in dict_df.items()]
all_df_att['threshold'] = [round(value.shape[0]/2) for key, value in dict_df.items()]
all_df_att['start_col'] = [value.shape[1] for key,value in dict_df.items()]
all_df_att


# In[49]:


#New Mexico 2007
null_counts = nm07.isnull().sum()
nulls = null_counts[null_counts>628]
sc_list = list(nulls.index)

#remove them
nm07 = nm07.drop(sc_list,axis=1)
all_df_att['end_col'] = nm07.shape[1]
all_df_att


# In[50]:


#New Mexico 2017
null_counts = nm17.isnull().sum()
nulls = null_counts[null_counts>482]
sc_list = list(nulls.index)

#remove them
nm17 = nm17.drop(sc_list,axis=1)
all_df_att.loc[all_df_att['name']=='nm17',['end_col']] = nm17.shape[1]
all_df_att


# In[51]:


#Nebraska 2007
null_counts = ne07.isnull().sum()
nulls = null_counts[null_counts>396]
sc_list = list(nulls.index)

#remove them
ne07 = ne07.drop(sc_list,axis=1)
all_df_att.loc[all_df_att['name']=='ne07',['end_col']] = ne07.shape[1]
all_df_att


# In[52]:


#Nebraska 2017
null_counts = ne17.isnull().sum()
nulls = null_counts[null_counts>447]
sc_list = list(nulls.index)

#remove them
ne17 = ne17.drop(sc_list,axis=1)
all_df_att.loc[all_df_att['name']=='ne17',['end_col']] = ne17.shape[1]
all_df_att


# In[53]:


### Imputing null values with mean ###
# use scikitlearn imputer to fill in values for the rest of the columns with null values by accessing Py_Scripts


# In[54]:


def impute_df(df):
    """Returns a dataframe with mean imputed values for NaN."""
    my_imputer = SimpleImputer(missing_values=np.nan)
    data_with_imputed_values = pd.DataFrame(my_imputer.fit_transform(df),columns = df.columns)
    return data_with_imputed_values


# In[55]:


nm07 = impute_df(nm07)
nm17 = impute_df(nm17)
ne07 = impute_df(ne07)
ne17 = impute_df(ne17)


# In[56]:


def only_zero(df):
    """Drops all columns that are all zero values and returns a Dataframe."""
    filter = pd.DataFrame(df.sum(axis=0)==0, columns=['value'])
    filter = filter.loc[filter['value']==True]
    col = list(filter.index)
    return df.drop(col,axis=1)


# In[57]:


nm07 = only_zero(nm07)
nm17 = only_zero(nm17)
ne07 = only_zero(ne07)
ne17 = only_zero(ne17)


# In[58]:


all_df_att['orig'] = 0
all_df_att['final_col']=0

all_df_att.loc[all_df_att['name']=='nm07',['orig']] = df_nm07.shape[1]
all_df_att.loc[all_df_att['name']=='nm17',['orig']] = df_nm17.shape[1]
all_df_att.loc[all_df_att['name']=='ne07',['orig']] = df_ne07.shape[1]
all_df_att.loc[all_df_att['name']=='ne17',['orig']] = df_ne17.shape[1]

all_df_att.loc[all_df_att['name']=='nm07',['final_col']] = nm07.shape[1]
all_df_att.loc[all_df_att['name']=='nm17',['final_col']] = nm17.shape[1]
all_df_att.loc[all_df_att['name']=='ne07',['final_col']] = ne07.shape[1]
all_df_att.loc[all_df_att['name']=='ne17',['final_col']] = ne17.shape[1]


# In[59]:


all_df_att.set_index('name')


# In[60]:


quarter = (all_df_att['orig'].mean())*.25

fig, ax = plt.subplots(figsize=(20,10))
all_df_att[['orig','final_col']].plot.bar(ax=ax)
plt.xlabel('Dataset',fontsize=15)
plt.ylabel('Column count',fontsize=15)
plt.xticks([0, 1, 2,3],labels=['nm07','nm17','ne07','ne17'],rotation=360)
plt.axhline(y=quarter,linewidth=1,color='r')
plt.legend(fontsize=15)
plt.title('We are left with a quarter of the original columns \n (red line shows the quarter mark of the original column mean count)',fontsize=20)
plt.savefig('C:/Users/Casey/Desktop/SNAP/Images/final_null.png');


# In[61]:


nm07_target.reset_index(drop=True,inplace=True)
nm07 = nm07.assign(CAT_ELIG=nm07_target)
nm07 = nm07.astype('float64')
nm07.to_csv('C:/Users/Casey/Desktop/SNAP/Data/clean_nm07.csv',index=None)


# In[62]:


nm17_target.reset_index(drop=True,inplace=True)
nm17 = nm17.assign(CAT_ELIG=nm17_target)
nm17 = nm17.astype('float64')
nm17.to_csv('C:/Users/Casey/Desktop/SNAP/Data/clean_nm17.csv',index=None)


# In[63]:


ne07_target.reset_index(drop=True,inplace=True)
ne07 = ne07.assign(CAT_ELIG=ne07_target)
ne07 = ne07.astype('float64')
ne07.to_csv('C:/Users/Casey/Desktop/SNAP/Data/clean_ne07.csv',index=None)


# In[64]:


ne17_target.reset_index(drop=True,inplace=True)
ne17 = ne17.assign(CAT_ELIG=ne17_target)
ne17 = ne17.astype('float64')
ne17.to_csv('C:/Users/Casey/Desktop/SNAP/Data/clean_ne17.csv',index=None)


# In[ ]:




