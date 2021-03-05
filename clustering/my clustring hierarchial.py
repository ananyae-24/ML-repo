import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values
import scipy.cluster.hierarchy as sci
dendogram=sci.dendrogram(sci.linkage(X,method="ward"))
plt.show()
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=3)
ans=hc.fit_predict(X)
plt.scatter(X[ans==0,0],X[ans==0,1],color="red")
plt.scatter(X[ans==1,0],X[ans==1,1],color="black")
plt.scatter(X[ans==2,0],X[ans==2,1],color="blue")
#plt.scatter(X[ans==3,0],X[ans==3,1],color="green")
#plt.scatter(X[ans==4,0],X[ans==4,1],color="purple")
plt.show()
