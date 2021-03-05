import numpy as np
from numpy import linalg as LA
class pca:
    def __init__(A,train):
        temp=list()
        for i in train:
            temp.append([i[0],i[1],i[2],i[3]])
        A.train=np.array(temp)
##        xT=np.transpose(x)
    def cal_varience(A):
        mean1=sum(A.train[:,0])/len(A.train[:,0])
        mean2=sum(A.train[:,1])/len(A.train[:,0])
        mean3=sum(A.train[:,2])/len(A.train[:,0])
        mean4=sum(A.train[:,3])/len(A.train[:,0])
        su=np.zeros((4,4))
        print(su)
        A.mean=np.array([mean1,mean2,mean3,mean4])
        for i in range(len(A.train[:,0])):
            A.train[i]=A.train[i]-A.mean
            x=np.array(A.train[i])
            x=x.transpose()
##            print(x)
            y=A.train[i].dot(x)
            print(y)
            su+=y
        A.varience=su
        
    def cal_eig(A):
        w,v=LA.eig(A.varience)
    def show(A):
        A.cal_varience()
        print(A.varience)
        
    
