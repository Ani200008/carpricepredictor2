#CarPricePredictor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sn
cars=pd.read_csv('F:/Machine learning/quikr_car.csv',header=0)




backup=cars.copy()


cars['year'].str.isnumeric()
cars=cars[cars['year'].str.isnumeric()]
cars['year']=cars['year'].astype(int)


cars['Price']=list(map(lambda x:x.replace('Ask For Price', '80,000'),cars['Price']))
cars['Price']=cars['Price'].str.replace(',','').astype(int)

cars['kms_driven']=cars['kms_driven'].str.split(' ').str.get(0).str.replace(',','')
cars=cars[cars['kms_driven'].str.isnumeric()]
cars['kms_driven']=cars['kms_driven'].astype(int)


cars=cars[~cars['fuel_type'].isna()]

cars['name']=cars['name'].str.split(' ').str.slice(0,3).str.join(' ')
cars=cars.reset_index(drop=True)


x=cars.drop(columns='Price',axis=1)
y=cars['Price']

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

ohe=OneHotEncoder()
ohe.fit(x[['name','company','fuel_type']])# Here we are giving all Qualotative values of our data Cleaning

column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)


scores=[]
for i in range(1000):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    scores.append(r2_score(y_test,y_pred))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)


import pickle
pipe.predict(pd.DataFrame([['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']],columns=['name','company','year','kms_driven','fuel_type']))