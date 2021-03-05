import math
class NB:
    def __init__(A,training,test):
        A.train=training
        A.test=test
        A.final=[]
    def caldensity(A,mean,standard,x):
        expp=math.exp(-1*((x-mean)**2)/(2*(standard**2)))
##        print(expp)
        anss=expp/(math.sqrt(2*math.pi)*standard)
##        print(anss)
        return anss
    def prob(A,entry,m_avg):
        p=1
        for i in range(len(entry)):
##            print(m_avg[i][0],m_avg,entry[i])
            p*=A.caldensity(m_avg[i][0],m_avg[i][1],entry[i])
        return p
            
    def cal_mean(A,l):
        mean=sum(l)/len(l)
        l2=[]
        for i in range(len(l)):
            l2.append(l[i]**2)
        sd=math.sqrt((sum(l2)/len(l2)-mean**2))
        return (mean,sd)
    def sort(A):
        d=dict()
        for i in A.train:
            if i[4] in d.keys():
                d[i[4]].append(i[:4])
            else:
                d[i[4]]=list()
        A.sep=d
    def classmean(A):
        c=dict()
        for i in A.sep:
            l=A.sep[i]
            y1=list()
            y2=list()
            y3=list()
            y4=list()
            for j in l:
                y1.append(j[0])
                y2.append(j[1])
                y3.append(j[2])
                y4.append(j[3])
            c[i]=[A.cal_mean(y1)]
            c[i].append(A.cal_mean(y2))
            c[i].append(A.cal_mean(y3))
            c[i].append(A.cal_mean(y4))
        A.classavg=c
    def overallmean(A):
        y1=list()
        y2=list()
        y3=list()
        y4=list()
        for j in A.train:
            y1.append(j[0])
            y2.append(j[1])
            y3.append(j[2])
            y4.append(j[3])
        A.wholeavg=[A.cal_mean(y1),A.cal_mean(y2),A.cal_mean(y3),A.cal_mean(y4)]
    def cal(A):
        for i in A.test:
            ans=[0,0]
            ans[0]=i[4]
            p=0
            for j in A.classavg:
                t=A.prob(i[:4],A.classavg[j])*(len(A.sep[i[4]])/len(A.train))/A.prob(i[:4],A.wholeavg)
                if t>p:
                    p=t
                    ans[1]=j
            A.final.append(ans)           
              
    def show(A):
        #A.caldensity(
        A.sort()
        A.classmean()
        A.overallmean()
        A.cal()
        return A.final
                
                

