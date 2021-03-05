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
from sklearn.neighbors import KNeighborsClassifier
classfier=KNeighborsClassifier(metric="minkowski",p=2);
classfier.fit(X_train,y_train)
ans=classfier.predict(X_test)
final=np.concatenate((np.reshape(y_test,(len(y_test),1)),np.reshape(ans,(len(y_test),1))),1)
from sklearn.metrics import confusion_matrix,accuracy_score
#confusion_matrix(y_test,ans)
#accuracy_score(y_test,ans)
print(confusion_matrix(y_test,ans),"\n",accuracy_score(y_test,ans))


