import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:,1:].values
y=dataset.iloc[:,3:].values
#print(y)
from sklearn.cluster import KMeans
wss=list()
for i in range(1,11):
    km=KMeans(n_clusters=i,init="k-means++",random_state=42)
    km.fit(y)
    wss.append(km.inertia_)
#plt.plot(range(1,11),wss,color="red")
km=KMeans(n_clusters=5,init="k-means++",random_state=42)
ans=km.fit_predict(y)
#print(ans)
plt.scatter(y[ans==0,0],y[ans==0,1],color="red")
plt.scatter(y[ans==1,0],y[ans==1,1],color="black")
plt.scatter(y[ans==2,0],y[ans==2,1],color="blue")
plt.scatter(y[ans==3,0],y[ans==3,1],color="green")
plt.scatter(y[ans==4,0],y[ans==4,1],color="purple")
plt.show()
