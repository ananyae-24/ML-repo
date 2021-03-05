import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Social_Network_Ads.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.svm import SVC
svc=SVC(kernel="rbf")
svc.fit(X_train, y_train)
ans=svc.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,ans),"\n",accuracy_score(y_test,ans))


