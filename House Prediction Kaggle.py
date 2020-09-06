#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv(r'C:\Users\LENOVO\Downloads\train.csv')


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# In[5]:


data.info()


# In[6]:


data.shape


# In[7]:


data['LotFrontage']=data['LotFrontage'].fillna(data['LotFrontage'].mean())


# In[8]:


data=data.drop(columns=['Alley','PoolQC','Fence','MiscFeature','FireplaceQu'])


# In[9]:


data['MasVnrType']=data['MasVnrType'].fillna(data['MasVnrType'].mode()[0])
data['MasVnrArea']=data['MasVnrArea'].fillna(data['MasVnrArea'].mean())
data['BsmtQual']=data['BsmtQual'].fillna(data['BsmtQual'].mode()[0])
data['BsmtCond']=data['BsmtCond'].fillna(data['BsmtCond'].mode()[0])
data['BsmtExposure']=data['BsmtExposure'].fillna(data['BsmtExposure'].mode()[0])
data['BsmtFinType1']=data['BsmtFinType1'].fillna(data['BsmtFinType1'].mode()[0])
data['BsmtFinType2']=data['BsmtFinType2'].fillna(data['BsmtFinType2'].mode()[0])
data['GarageType']=data['GarageType'].fillna(data['GarageType'].mode()[0])
data['GarageYrBlt']=data['GarageYrBlt'].fillna(data['GarageYrBlt'].mean())
data['GarageFinish']=data['GarageFinish'].fillna(data['GarageFinish'].mode()[0])
data['GarageQual']=data['GarageQual'].fillna(data['GarageQual'].mode()[0])
data['GarageCond']=data['GarageCond'].fillna(data['GarageCond'].mode()[0])


# In[10]:


data.info()


# In[11]:


data=data.dropna()


# In[12]:


data.shape


# In[13]:


data=data.drop(columns=['Id'])


# In[14]:


test_data=pd.read_csv('Test_Data.csv')


# In[15]:


test_data.head()


# In[16]:


test_data.shape


# In[17]:


new_data=pd.concat([data,test_data],axis=0,ignore_index=True)


# In[18]:


new_data.shape


# In[19]:


new_data.head()


# In[20]:


print(new_data.info())


# In[21]:


col=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2',
    'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','BsmtQual',
    'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Foundation','Electrical','KitchenQual',
    'Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition']


# In[22]:


len(col)


# In[23]:


for i in range(len(col)):
    print(new_data[col[i]].unique())


# In[24]:


for i in range(len(col)):
    dummie=pd.get_dummies(new_data[col[i]])
    print(dummie)
    print(new_data.shape)
    new_data=pd.concat([new_data,dummie],axis=1)
    print(new_data)
    new_data=new_data.drop(columns=[col[i]],axis=1)


# In[25]:


new_data.shape


# In[26]:


new_data.select_dtypes(include=['object'])


# In[27]:


new_data=new_data.loc[:,~new_data.columns.duplicated()]


# In[28]:


new_data.shape


# In[29]:


train=new_data.iloc[:1449,:]
test=new_data.iloc[1449:,:]


# In[30]:


test.drop(['SalePrice'],axis=1,inplace=True)


# In[31]:


X_train=train.drop(['SalePrice'],axis=1)
y_train=train['SalePrice']


# In[32]:


X_train.info()


# In[33]:


y_train.unique


# In[34]:


from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(test)


# In[35]:


sample_file=pd.read_csv(r'C:\Users\LENOVO\Downloads\sample_submission.csv')


# In[36]:


sample_file.shape


# In[37]:


test.shape


# In[38]:


submission=pd.DataFrame({'Id':sample_file['Id'],'SalePrice':y_pred})
submission.head()


# In[39]:


filename = 'House Prediction Submmision.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)


# In[ ]:




