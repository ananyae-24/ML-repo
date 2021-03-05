import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=.2,random_state=1)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train, y_train)
y_pred=lr.predict(X_test)
plt.scatter(X_train,y_train,color="red")
plt.scatter(X_test,y_test,color="green")
plt.plot(X_train,lr.predict(X_train),color="blue")
plt.title("Salary vs experience")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()
