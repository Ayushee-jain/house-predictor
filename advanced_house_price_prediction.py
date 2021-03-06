# -*- coding: utf-8 -*-
"""Advanced_House_price_prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16zibS82riCQifpND4ucK6HVcdcQBgKRN
"""

! pip install -q kaggle

from google.colab import files

files.upload()

! mkdir ~/.kaggle

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets list

! kaggle competitions download -c 'house-prices-advanced-regression-techniques'

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train=pd.read_csv('/content/train.csv')
test=pd.read_csv('/content/test.csv')

train.shape

data=pd.concat([train,test])

data.head()

data.shape

x_data=data.drop(columns=['SalePrice'])
y_data=data['SalePrice']

a=x_data.isnull().sum()

print(a)

cols=x_data.columns

print(cols)

for i in range(0,len(a)):
  if a[i]!=0:
    print(x_data[cols[i]].dtypes,a[i],i)

x_data[cols[3]]=x_data[cols[3]].replace(np.NaN,x_data[cols[3]].mean())
x_data[cols[26]]=x_data[cols[26]].replace(np.NaN,x_data[cols[26]].mean())
x_data[cols[34]]=x_data[cols[34]].replace(np.NaN,x_data[cols[34]].mean())
x_data[cols[36]]=x_data[cols[36]].replace(np.NaN,x_data[cols[36]].mean())
x_data[cols[37]]=x_data[cols[37]].replace(np.NaN,x_data[cols[37]].mean())
x_data[cols[38]]=x_data[cols[38]].replace(np.NaN,x_data[cols[38]].mean())
x_data[cols[47]]=x_data[cols[47]].replace(np.NaN,x_data[cols[47]].mean())
x_data[cols[48]]=x_data[cols[48]].replace(np.NaN,x_data[cols[48]].mean())
x_data[cols[59]]=x_data[cols[59]].replace(np.NaN,x_data[cols[59]].mean())
x_data[cols[61]]=x_data[cols[61]].replace(np.NaN,x_data[cols[61]].mean())
x_data[cols[62]]=x_data[cols[62]].replace(np.NaN,x_data[cols[62]].mean())

a=x_data.isnull().sum()
for i in range(0,len(a)):
  if a[i]!=0:
    print(x_data[cols[i]].dtypes,i,a[i])

print(cols[72],cols[74],cols[6],cols[73])

x_data=x_data.drop(columns=['PoolQC','MiscFeature','Alley','Fence'],axis=1)

cols=x_data.columns

a=x_data.isnull().sum()
for i in range(0,len(a)):
  if a[i]!=0:
    print(x_data[cols[i]].dtypes,i,a[i])

import seaborn as sns
sns.countplot(x_data[cols[2]])

x_data[cols[2]]=x_data[cols[2]].replace(np.NaN,'RL')

sns.countplot(x_data[cols[8]])

x_data[cols[8]]=x_data[cols[8]].replace(np.NaN,'AllPub')

sns.countplot(x_data[cols[22]])

x_data[cols[22]]=x_data[cols[22]].replace(np.NaN,'VinylSd')

sns.countplot(x_data[cols[23]])

x_data[cols[23]]=x_data[cols[23]].replace(np.NaN,'VinylSd')

sns.countplot(x_data[cols[24]])

x_data[cols[24]]=x_data[cols[24]].replace(np.NaN,'None')

sns.countplot(x_data[cols[29]])

x_data[cols[29]]=x_data[cols[29]].replace(np.NaN,'TA')

sns.countplot(x_data[cols[30]])

x_data[cols[30]]=x_data[cols[30]].replace(np.NaN,'TA')

sns.countplot(x_data[cols[31]])

x_data[cols[31]]=x_data[cols[31]].replace(np.NaN,'No')

sns.countplot(x_data[cols[32]])

x_data[cols[32]]=x_data[cols[32]].replace(np.NaN,'GLQ')

sns.countplot(x_data[cols[34]])

x_data[cols[34]]=x_data[cols[34]].replace(np.NaN,'Unf')

sns.countplot(x_data[cols[41]])

x_data[cols[41]]=x_data[cols[41]].replace(np.NaN,'SBrkr')

sns.countplot(x_data[cols[52]])

x_data[cols[52]]=x_data[cols[52]].replace(np.NaN,'TA')

sns.countplot(x_data[cols[54]])

x_data[cols[54]]=x_data[cols[54]].replace(np.NaN,'Typ')

sns.countplot(x_data[cols[56]])

x_data[cols[56]]=x_data[cols[56]].replace(np.NaN,'New')

sns.countplot(x_data[cols[57]])

x_data[cols[57]]=x_data[cols[57]].replace(np.NaN,'Attchd')

sns.countplot(x_data[cols[59]])

x_data[cols[59]]=x_data[cols[59]].replace(np.NaN,'unf')

sns.countplot(x_data[cols[62]])

x_data[cols[62]]=x_data[cols[62]].replace(np.NaN,'TA')

sns.countplot(x_data[cols[63]])

x_data[cols[63]]=x_data[cols[63]].replace(np.NaN,'TA')

sns.countplot(x_data[cols[74]])

x_data[cols[74]]=x_data[cols[74]].replace(np.NaN,'WD')

a=x_data.isnull().sum()
for i in range(0,len(a)):
  if a[i]!=0:
    print(x_data[cols[i]].dtypes,i,a[i])

x_data.shape

x_data.corr()

new_data=pd.DataFrame({'Id':x_data['Id'],'MSSubClass':x_data['MSSubClass'],'LotFrontage':x_data['LotFrontage'],'LotArea':x_data['LotArea'],'OverallQual':x_data['OverallQual'],'OverallCond':x_data['OverallCond'],
                       'YearBuilt':x_data['YearBuilt'],'YearRemodAdd':x_data['YearRemodAdd'],'MasVnrArea':x_data['MasVnrArea'],'BsmtFinSF1':x_data['BsmtFinSF1'],'BsmtFinSF2':x_data['BsmtFinSF2'],
                       'BsmtUnfSF':x_data['BsmtUnfSF'],'TotalBsmtSF':x_data['TotalBsmtSF'],'1stFlrSF':x_data['1stFlrSF'],'2ndFlrSF':x_data['2ndFlrSF'],'LowQualFinSF':x_data['LowQualFinSF'],'GrLivArea':x_data['GrLivArea'],
                       'BsmtFullBath':x_data['BsmtFullBath'],'BsmtHalfBath':x_data['BsmtHalfBath'],'FullBath':x_data['FullBath'],'HalfBath':x_data['HalfBath'],'BedroomAbvGr':x_data['BedroomAbvGr'],
                       'KitchenAbvGr':x_data['KitchenAbvGr'],'TotRmsAbvGrd':x_data['TotRmsAbvGrd'],'Fireplaces':x_data['Fireplaces'],'GarageYrBlt':x_data['GarageYrBlt'],
                       'GarageCars':x_data['GarageCars'],'GarageArea':x_data['GarageArea'],'WoodDeckSF':x_data['WoodDeckSF'],'OpenPorchSF':x_data['OpenPorchSF'],
                       'EnclosedPorch':x_data['EnclosedPorch'],'3SsnPorch':x_data['3SsnPorch'],'ScreenPorch':x_data['ScreenPorch'],'PoolArea':x_data['PoolArea'],
                       'MoSold':x_data['MoSold'],'YrSold':x_data['YrSold']})

x_data=x_data.drop(columns=['GarageArea','1stFlrSF','TotRmsAbvGrd'])

cols=x_data.columns
for i in range(0,len(cols)):
  if(x_data[cols[i]].dtypes=='object'):
    a=pd.get_dummies(x_data[cols[i]],drop_first=True)
    new_data=pd.concat([new_data,a],axis=1)

new_data.shape

new_data=new_data.drop(columns=['Id'])

x_train=new_data[:1460]
x_test=new_data[1460:]

y_train=y_data[:1460]

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

sample_file=pd.read_csv('/content/sample_submission.csv')

sample_file.shape

submission=pd.DataFrame({'Id':sample_file['Id'],'SalePrice':y_pred})
submission.head()

filename = 'House Prediction Submmision.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)

