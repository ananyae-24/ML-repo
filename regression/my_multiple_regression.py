import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv("50_Startups.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[-1])],remainder="passthrough")
X=np.array(ct.fit_transform(X))
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test=train_test_split(X,y,test_size=.2,random_state=1)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train, y_train)
y_pred=lr.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_pred),1)),1))
print(lr.coef_)
print(lr.intercept_)