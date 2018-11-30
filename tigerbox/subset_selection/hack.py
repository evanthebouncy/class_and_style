import numpy as np
import random
import time
import pandas as pd
#import numpy as np
#from plotBoundary import *
import pylab as pl
import math
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from keras.datasets import mnist


L = 10
NOISE = 0.02
N_PROTO = 20

# fix the seed
np.random.seed(1)
random.seed(1)

# generate a structure vector
def gen_struct(n_proto, l=L):
  return [np.random.randint(0, 2, size=(l,)) for _ in range(n_proto)]

# a simple 2 klass problem with pre-defined klass and style
def gen_klass(n_proto):
  klass1 = gen_struct(n_proto)
  klass2 = gen_struct(n_proto)
  return klass1, klass2

# make a vector noisey
def apply_noise(x):
  x = np.copy(x)
  for i in range(len(x)):
    if random.random() < NOISE:
      x[i] = 1 - x[i]
  return x

def disjoint(kl1s, kl2s):
  kl1s = set([str(x) for x in kl1s])
  kl2s = set([str(x) for x in kl2s])
  return len(kl1s.intersection(kl2s)) == 0

def make_dataset(n):
  kl1s, kl2s = gen_klass(N_PROTO)
  assert disjoint(kl1s, kl2s)
  commons = gen_struct(N_PROTO)
  X = []
  Y = []
  for i in range(n):
    toss = random.random() < 0.5
    kl = random.choice(kl1s) if toss else random.choice(kl2s)
    common = random.choice(commons)
    signal = np.concatenate((kl,
                             common,
                             ))
    signal = apply_noise(signal)
    X.append(signal)
    Y.append(0 if toss else 1)
  return np.array(X), np.array(Y)

def gen_data(n):
  n_tr = n // 2

  X, Y = make_dataset(n)
  X_tr, Y_tr = X[:n_tr], Y[:n_tr]
  X_t, Y_t = X[n_tr:], Y[n_tr:]

  return X_tr, Y_tr, X_t, Y_t



class KNN:
    def __init__(self,K):
        self.name="KNN"
        self.K=K
    def getRef(self,X,Y):
        self.X=X
        self.Y=Y
    def dist(self,x1,x2):
        return np.linalg.norm(x2-x1)
    def predict(self,x):
        dis=np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            dis[i]=self.dist(x,self.X[i])
        k=np.argsort(dis)
        ans=np.zeros(10)
        for i in range(self.K):
                ans[self.Y[k[i]]]=ans[self.Y[k[i]]]+1
        return np.argmax(ans)



    def eval(self,testX,testY):
        if(self.Y.shape[0]==0):
            return 0.0
        ans=0.0
        for i in range(testX.shape[0]):
            #print("have",i)
            #print(self.dist(X[437],X[499]))
            if(self.predict(testX[i])==testY[i]):
                ans=ans+1
            #print(self.predict(testX[i]),testY[i])
        return ans/float(testX.shape[0])

    def goodpoints(self):
        self.getRef(X,Y)
        good=np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            if(self.predict(X[i])==Y[i]):
                good[i]=1
        return (X[good==1],Y[good==1])

    def count(self,x,y):
        dis = np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            dis[i] = self.dist(x, self.X[i])
        k = np.argsort(dis)
        ans = 0
        tt = 0
        for i in range(self.X.shape[0]):
            #ans[self.Y[k[i]]] = ans[self.Y[k[i]]] + 1
            if(self.Y[k[i]]==y):
                ans=ans+((5 - dis[k[i]]) / 10)
                tt = tt + 1
                if(tt>=self.K):
                    break
            else:
                ans = ans - ((5 - dis[k[i]]) / 10)

        return ans

    def loss(self,testX,testY): #larger the better
        #return self.eval(testX,testY)
        ans = 0.0
        for i in range(testX.shape[0]):
            # print("have",i)
            # print(self.dist(X[437],X[499]))
            #if (self.predict(testX[i]) == testY[i]):
            #    ans = ans + 1

            #ans=ans+math.log(self.count(testX[i],testY[i]))
            ans = ans + (self.count(testX[i], testY[i]))
            # print(self.predict(testX[i]),testY[i])
        return ans

class KNN_wei:
    def __init__(self,K):
        self.name="KNN"
        self.K=K
    def getRef(self,X,Y):
        self.X=X
        self.Y=Y
        self.wei=np.zeros(Y.shape)
    def dist(self,x1,x2):
        return np.linalg.norm(x2-x1)
    def predict(self,x):
        dis=np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            dis[i]=self.dist(x,self.X[i])
        k=np.argsort(dis)
        ans=np.zeros(10)
        for i in range(self.K):
                ans[self.Y[k[i]]]=ans[self.Y[k[i]]]+1
        return np.argmax(ans)

    def assignwei(self,x,y,wei):
        dis=np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            dis[i]=self.dist(x,self.X[i])
        k=np.argsort(dis)
        if(self.Y[k[0]]==y):
            self.wei[k[0]]=self.wei[k[0]]+wei
        else:
            self.wei[k[0]] = self.wei[k[0]] - wei




    def eval(self,testX,testY,wei):
        if(self.Y.shape[0]==0):
            return 0.0
        ans=0.0
        for i in range(testX.shape[0]):
            #print("have",i)
            #print(self.dist(X[437],X[499]))
            if(self.predict(testX[i])==testY[i]):
                ans=ans+wei[i]
            #print(self.predict(testX[i]),testY[i])
        return ans/float(testX.shape[0])

    def getwei(self,testX,testY,wei):
        if (self.Y.shape[0] == 0):
            return []
        ans = 0.0
        for i in range(testX.shape[0]):
            # print("have",i)
            # print(self.dist(X[437],X[499]))
            self.assignwei(testX[i],testY[i],wei[i])
            # print(self.predict(testX[i]),testY[i])
        return self.wei

    def goodpoints(self):
        self.getRef(X,Y)
        good=np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            if(self.predict(X[i])==Y[i]):
                good[i]=1
        return (X[good==1],Y[good==1])

    def count(self,x,y):
        dis = np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            dis[i] = self.dist(x, self.X[i])
        k = np.argsort(dis)
        ans = 0
        tt = 0
        for i in range(self.X.shape[0]):
            #ans[self.Y[k[i]]] = ans[self.Y[k[i]]] + 1
            if(self.Y[k[i]]==y):
                ans=ans+((5 - dis[k[i]]) / 10)
                tt = tt + 1
                if(tt>=self.K):
                    break
            else:
                ans = ans - ((5 - dis[k[i]]) / 10)

        return ans

    def loss(self,testX,testY): #larger the better
        #return self.eval(testX,testY)
        ans = 0.0
        for i in range(testX.shape[0]):
            # print("have",i)
            # print(self.dist(X[437],X[499]))
            #if (self.predict(testX[i]) == testY[i]):
            #    ans = ans + 1

            #ans=ans+math.log(self.count(testX[i],testY[i]))
            ans = ans + (self.count(testX[i], testY[i]))
            # print(self.predict(testX[i]),testY[i])
        return ans






def evalModel(X,Y,testX,testY,k=3):
    knn=KNN(k)
    knn.getRef(X,Y)
    return knn.eval(testX,testY)


def evalModel_wei(X,Y,testX,testY,k,wei):
    knn=KNN_wei(k)
    knn.getRef(X,Y)
    return knn.eval(testX,testY,wei)

def getwei(X,Y,testX,testY,k,wei):
    knn=KNN_wei(k)
    knn.getRef(X,Y)
    return knn.getwei(testX,testY,wei)


def sub_select_knn_wei(X, Y, n_samples,wei):
    from copy import deepcopy
    data_size = len(X)

    def random_other_idx(sub_idxs):
        ret = random.randint(0, data_size - 1)
        if ret in sub_idxs:
            return random_other_idx(sub_idxs)
        return ret

    def swap_one(sub_idxs, swap_idx):
        sub_idxs = deepcopy(sub_idxs)
        sub_idxs[swap_idx] = random_other_idx(sub_idxs)
        return sub_idxs

    def get_subset_score(X_sub, Y_sub):
        return evalModel_wei(X_sub, Y_sub, X, Y, KNNpoint,wei)


    sub_idxs = list(range(n_samples))
    X_sub, Y_sub = X[sub_idxs, :], Y[sub_idxs]

    score_old = get_subset_score(X_sub, Y_sub)
    stuck_cnt = 0
    for i in range(n_samples * 100):
        stuck_cnt += 1
        #if (i % 10 == 0):
            #print("new score!", getscore(X_sub, Y_sub))
        if stuck_cnt > n_samples:
            break
        swap_idx = i % n_samples
        new_sub_idxs = swap_one(sub_idxs, swap_idx)
        new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]

        score_new = get_subset_score(new_X_sub, new_Y_sub)

        if score_new > score_old:
            #print("[knn_anneal] iteration ", i, "stuck ", stuck_cnt, " new score! : ", score_new)
            X_sub, Y_sub = new_X_sub, new_Y_sub
            sub_idxs = new_sub_idxs
            score_old = score_new
            stuck_cnt = 0



    return X[sub_idxs, :], Y[sub_idxs], getwei(X[sub_idxs, :], Y[sub_idxs],X,Y,KNNpoint,wei)




def subluangao(X,Y,leng,ratio,wei):
    if(X.shape[0]<=leng):
        return sub_select_knn_wei(X,Y,int(X.shape[0]*ratio),wei)
    else:
        k=np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            k[i]=np.var(X[:,i])

        index = np.argmax(k)
        mid = int(X.shape[0]/2)
        maxi = 0.0
        sor = np.argsort(X[:,index])

        k=math.sqrt(X.shape[0])/2
        k=int(k)

        for i in range(k,X.shape[0]-k):

            if(X[sor[i+k],index]-X[sor[i-k],index]>maxi):
                maxi = X[sor[i+k],index]-X[sor[i-k],index]
                mid = i

        left = sor[:mid]
        right = sor[mid:]
        Xleft,Yleft,weileft = subluangao(X[left],Y[left],leng,ratio,wei[left])
        Xright,Yright,weiright = subluangao(X[right],Y[right],leng,ratio,wei[right])

        return np.append(Xleft,Xright,axis=0),np.append(Yleft,Yright,axis=0),np.append(weileft,weiright,axis=0)

def luangao2(leng,ratio,chosepoint,X,Y,t=10000):

    rchosepoint = chosepoint
    from copy import deepcopy
    wei = np.ones(Y.shape,float)
    print('hhh',X.shape[0])
    for i in range(t):
        print(i)

        X,Y,wei=subluangao(X,Y,leng,ratio,wei)
        print('hhh',X.shape[0])
        if(X.shape[0]*ratio*0.5<chosepoint):
            break

    Y=Y.astype(int)
    xx,yy,ww= sub_select_knn_wei(X,Y,rchosepoint,wei)
    return xx,yy





if __name__ == '__main__':
  dat=2000
  X,Y = make_dataset(dat)
  print(X.shape,Y.shape)

  K = pd.read_pickle('mnist_dim32.p')
  print(K.shape)
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  print(y_train.shape,y_test.shape)
  X=K
  Y=y_train

  XX=deepcopy(X)
  YY=deepcopy(Y)
  together = zip(X,Y)
  chosepoint=100
  KNNpoint=1
  batchsize = 10
  ratio = 0.5
  dat_size = X.shape[0]

  starttime = time.time()
o
  xt1,yt1=luangao2(batchsize,ratio,chosepoint,X,Y)
  print(time.time()-starttime)

  print("to beat updated", evalModel(xt1, yt1, XX, YY, KNNpoint))
