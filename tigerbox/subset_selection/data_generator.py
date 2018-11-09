import numpy as np
import random
import time
#import numpy as np
#from plotBoundary import *
import pylab as pl
import math
from copy import deepcopy
from sklearn.linear_model import LogisticRegression


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


class SVMSKT():
    """Implementation of SVM with SGD with PEGASOS Algorithm"""

    def __init__(self):
        from sklearn import svm
        self.clf = svm.SVC(gamma='scale')

    def Ker(self,x1,x2):
        #return np.dot(x1,x2)
        return math.exp(-self.beta1*np.linalg.norm(x1-x2)*np.linalg.norm(x1-x2))


    def fit(self, X, Y):
        self.X=X
        self.Y=Y
        self.clf.fit(X,Y)

    def test(self,X,Y):
        #Y=list(Y)
        tt=np.sum(Y==self.clf.predict(X))

        return tt/(1.0*X.shape[0])


    def top(self,k,ZP=True):
        self.alpha=np.abs(self.clf.dual_coef_)[0]
        #print(self.alpha)
        index=np.argsort((-self.alpha))
        ax = np.zeros((k, self.X.shape[1]), dtype=int)
        ay = np.zeros(k, dtype=int)
        # print(X[index2[0]])
        for i in range(k):
            ax[i] = self.X[index[i]]
            ay[i] = self.Y[index[i]]
        if (ZP):
            return zip(ax, ay)
        else:
            return (ax, ay)

class LRegr:

  def __init__(self):
    self.name = "LRegr"

  def learn(self, train_corpus):
    train_data, train_label = train_corpus
    logisticRegr = LogisticRegression(solver='sag')
    logisticRegr.fit(train_data, train_label)
    self.logisticRegr = logisticRegr

  def evaluate(self, test_corpus):
    test_data, test_label = test_corpus
    label_pred = self.logisticRegr.predict(test_data)
    return np.sum(label_pred == test_label) / len(test_label)

  def top(self,k,ZP=True):
      #index=np.zeros(X.shape[0])
      #for i in range(X.shape[0]):
      index=np.abs(self.logisticRegr.predict_proba(X)[:,0]-0.5)
      #index=self.logisticRegr.predict_proba(X)
      #print(index[1,])

          #print(index[i])
      index2 = np.argsort(index)
      ax = np.zeros((k,X.shape[1]),dtype=int)
      ay = np.zeros(k,dtype=int)
      #print(X[index2[0]])
      for i in range(k):
          ax[i] = X[index2[i]]
          ay[i] = Y[index2[i]]
      if (ZP):
          return zip(ax, ay)
      else:
          return (ax, ay)

      #print(index.shape)



class Circle:
    def __init__(self):
        self.name="CircleAlg"


    def learn(self,XY):
        self.X,self.Y=XY
        self.score=np.zeros(self.X.shape[0])


    def dist(self,x1,x2):
        return np.linalg.norm(x2-x1)


    def top2(self,r):
        for i in range(self.X.shape[0]):
            cor=0.0
            tot=0.0
            for j in range(self.X.shape[0]):
                if(self.dist(self.X[i],self.X[j])<=r):
                    if(self.Y[i]==self.Y[j]):
                        cor=cor+1
                    tot=tot+1
            self.score[i]=(cor/tot-0.5)*tot
        inl=np.zeros(self.X.shape[0])
        rank=np.argsort(-self.score)
        ans = np.zeros(self.X.shape[0], dtype=int)
        #print(rank)
        for i in range(self.X.shape[0]):
            if(inl[rank[i]]==1):
                continue
            inl[rank[i]]=1

            ans[rank[i]]=1
            for j in range(self.X.shape[0]):
                if(self.dist(self.X[rank[i]],self.X[j])<=r):
                    if (self.Y[i] == self.Y[j]):
                        inl[j]=1
        return (self.X[ans==1], self.Y[ans==1])


    def top(self,k,ZP=True):
        l=0.0001
        r=1e9
        while(r-l>0.001):
            mid=(l+r)/2.0
            ansX,ansY=self.top2(mid)
            #print(ans)
            if(ansY.shape[0]>k):
                l=mid
            elif ansY.shape[0]<k:
                r=mid
            else:
                return ans
        ansX, ansY = self.top2(r)
        if(ZP):
            return zip(ansX,ansY)
        else:
            return (ansX,ansY)



        #return self.top2(100)

class KNNgraph:
    def __init__(self):
        self.name

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
        ans=np.zeros(2)
        for i in range(self.K):

                ans[self.Y[k[i]]]=ans[self.Y[k[i]]]+1
        return np.argmax(ans)
    def eval(self,testX,testY):
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





def SVM(k=10):
    svm = SVMSKT()

    #print(X.shape, Y.shape)
    svm.fit(X, Y)
    #print(svm.alpha)
    #print(svm.sv())
    print(svm.test(X, Y))
    #together=svm.top(k)
    #for zz in together:
    #    print (zz[1], zz[0])
    return svm.top(k,False)

def LGR(k=10):
    lregr = LRegr()
    lregr.learn((X, Y))
    print(lregr.evaluate((X, Y)))
    print(X.shape, Y.shape)
    together=lregr.top(k)
    for zz in together:
        print (zz[1], zz[0])
    return lregr.top(k,False)

def CCL(k=10):
    ccl = Circle()
    ccl.learn((X, Y))
    #a,b=zip(*together)
    #print(a,b)
    return ccl.top(k,False)

def RND(k=10):
    return (X[20:k+20,],Y[20:k+20])


def evalModel(X,Y,testX,testY,k=3):
    knn=KNN(k)
    knn.getRef(X,Y)
    return knn.eval(testX,testY)

def KNNLoss(X,Y,testX,testY,k=3):
    knn=KNN(k)
    knn.getRef(X,Y)
    return knn.loss(testX,testY)
def RNDtest(k=10,KNN=3):
    ans=0.0
    for i in range(10):
        st=i*10
        ans=ans+evalModel(X[st:st+k,],Y[st:st+k],X,Y,KNN)
    return ans/10.0
    #return (X[20:k+20,],Y[20:k+20])


# optimize knn by annealing, respecting the labels
def sub_select_knn(X, Y, n_samples):
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
        #return KNNLoss(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)
        return evalModel(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)

    def getscore(X_sub, Y_sub):
        #return KNNLoss(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)
        return evalModel(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)


    sub_idxs = list(range(n_samples))
    X_sub, Y_sub = X[sub_idxs, :], Y[sub_idxs]

    score_old = get_subset_score(X_sub, Y_sub)
    stuck_cnt = 0
    for i in range(n_samples * 100):
        stuck_cnt += 1
        if (i % 10 == 0):
            print("new score!", getscore(X_sub, Y_sub))
        if stuck_cnt > n_samples:
            break
        swap_idx = i % n_samples
        new_sub_idxs = swap_one(sub_idxs, swap_idx)
        new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]

        score_new = get_subset_score(new_X_sub, new_Y_sub)

        if score_new > score_old:
            print("[knn_anneal] iteration ", i, "stuck ", stuck_cnt, " new score! : ", score_new)
            X_sub, Y_sub = new_X_sub, new_Y_sub
            sub_idxs = new_sub_idxs
            score_old = score_new
            stuck_cnt = 0

    return X[sub_idxs, :], Y[sub_idxs]




def sub_select_knn_MST_init(X, Y, n_samples):
    from copy import deepcopy
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import distance_matrix
    data_size = len(X)

    def MST():
        dm=distance_matrix(X, X)
        print(dm)
        mst = minimum_spanning_tree(dm)
        mst=mst.toarray().astype(int)
        for i in range(data_size):
            for j in range(data_size):
                mst[i,j]=max(mst[i,j],mst[j,i])
        print(mst)
        graph=[]
        for i in range(data_size):
            k=np.where(mst[i] > 0)
            graph.append(k)
        print(graph)
        return graph,mst


    def random_other_idx(sub_idxs):
        ret = random.randint(0, data_size - 1)
        if ret in sub_idxs:
            return random_other_idx(sub_idxs)
        return ret

    def swap_one(sub_idxs, swap_idx):
        sub_idxs = deepcopy(sub_idxs)
        sub_idxs[swap_idx] = random_other_idx(sub_idxs)
        return sub_idxs

    def swap_(sub_idxs, swap_idx,new_idx):
        sub_idxs = deepcopy(sub_idxs)
        sub_idxs[swap_idx] = new_idx
        return sub_idxs

    def del_one(sub_idxs, del_idx,st):
        sub_idxs = deepcopy(sub_idxs)
        x=np.array(range(st),dtype=int)
        #print(x)
        return sub_idxs[x!=int(del_idx)]

    def init():
        ans=np.array([],dtype=int)
        for i in range(n_samples*1000):
            topv=0
            topx=0
            topy=0
            for j in range(data_size):
                k=np.argmax(mst[j])

                if(mst[j,k]>topv):
                    topv=mst[j,k]
                    topx=j
                    topy=k
            ans=np.append(ans,topx)
            #ans=np.append(ans,topy)
            ans=np.unique(ans)
            mst[topx,topy]=0
            mst[topy,topx]=0
            if(ans.shape[0]==n_samples):
                break
        return ans




    def get_subset_score(X_sub, Y_sub):
        return KNNLoss(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)
        #return evalModel(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)

    def getscore(X_sub, Y_sub):
        #return KNNLoss(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)
        return evalModel(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)

    #sub_idxs = list(range(n_samples))

    def shrinkinit(st):
        sub_idxs = np.array(range(st))
        X_sub, Y_sub = X[sub_idxs, :], Y[sub_idxs]
        while(st>n_samples):
            score_old=-1e8
            perma_id=deepcopy(sub_idxs)
            for i in range(st):
                new_sub_idxs = del_one(perma_id, i, st)
                new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]
                score_new = get_subset_score(new_X_sub, new_Y_sub)
                if score_new > score_old:
                    print("[knn_anneal] iteration ", i, " new score! : ", score_new)
                    X_sub, Y_sub = new_X_sub, new_Y_sub
                    sub_idxs = new_sub_idxs
                    score_old = score_new
                    #stuck_cnt = 0
                    sw = 1
            st=st-1
            print(st)
        return sub_idxs




    g,mst=MST()
    #sub_idxs = init()

    sub_idxs=shrinkinit(30)
    print(sub_idxs)
    X_sub, Y_sub = X[sub_idxs, :], Y[sub_idxs]
    score_old = get_subset_score(X_sub, Y_sub)
    stuck_cnt = 0
    swap_idx = 0
    changed = 0
    for i in range(n_samples * 1000000):
        stuck_cnt += 1
        if(i%10==0):
            print("new score!",getscore(X_sub, Y_sub))
        if changed > 10:
            changed = 0
            for j in range(n_samples):
                #break
                # print(j)
                swap_idx = j
                while (True):
                    sw = 0
                    # node=sub_idxs[swap_idx]
                    for J in g[sub_idxs[swap_idx]]:
                        # print(j)
                        new_sub_idxs = swap_(sub_idxs, swap_idx, J[0])
                        new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]
                        score_new = get_subset_score(new_X_sub, new_Y_sub)
                        if score_new > score_old:
                            print("[knn_anneal] iteration ", i, "stuck ", 0, " new score! : ", score_new)
                            X_sub, Y_sub = new_X_sub, new_Y_sub
                            sub_idxs = new_sub_idxs
                            score_old = score_new
                            stuck_cnt = 0
                            sw = 1
                    if (sw == 0):
                        break
                    #break

        if stuck_cnt > n_samples*500:
            break
        swap_idx = i % n_samples
        maxscore=score_old

        while(False):
            sw=0
            # node=sub_idxs[swap_idx]
            for j in g[sub_idxs[swap_idx]]:
                # print(j)
                new_sub_idxs = swap_(sub_idxs, swap_idx, j[0])
                new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]
                score_new = get_subset_score(new_X_sub, new_Y_sub)
                if score_new > score_old:
                    print("[knn_anneal] iteration ", i, "stuck ", stuck_cnt, " new score! : ", score_new)
                    X_sub, Y_sub = new_X_sub, new_Y_sub
                    sub_idxs = new_sub_idxs
                    score_old = score_new
                    stuck_cnt = 0
                    sw=1
            if(sw==0):
                break


        for j in range(1):
            #break
            #print(j)
            new_sub_idxs=swap_one(sub_idxs,swap_idx)
            new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]
            score_new = get_subset_score(new_X_sub, new_Y_sub)
            if score_new > score_old:
                print("[knn_anneal] iteration ", i, "stuck ", stuck_cnt, " new score! : ", score_new)
                X_sub, Y_sub = new_X_sub, new_Y_sub
                sub_idxs = new_sub_idxs
                score_old = score_new
                stuck_cnt = 0
                changed = changed + 1
        #node=sub_idxs[swap_idx]


        if(stuck_cnt == 0):
            #swap_idx = j
            while (True):
                sw = 0
                # node=sub_idxs[swap_idx]
                for J in g[sub_idxs[swap_idx]]:
                    # print(j)
                    new_sub_idxs = swap_(sub_idxs, swap_idx, J[0])
                    new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]
                    score_new = get_subset_score(new_X_sub, new_Y_sub)
                    if score_new > score_old:
                        print("[knn_anneal] iteration ", i, "stuck ", stuck_cnt, " new score! : ", score_new)
                        X_sub, Y_sub = new_X_sub, new_Y_sub
                        sub_idxs = new_sub_idxs
                        score_old = score_new
                        stuck_cnt = 0
                        sw = 1
                if (sw == 0):
                    break




    return X[sub_idxs, :], Y[sub_idxs]


def sub_select_knn_MST_init_KNN(X, Y, n_samples):
    from copy import deepcopy
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import distance_matrix
    data_size = len(X)

    def MST(K=3):
        dm=distance_matrix(X, X)
        print(dm)
        mst = minimum_spanning_tree(dm)
        mst=mst.toarray().astype(int)
        for i in range(data_size):
            for j in range(data_size):
                mst[i,j]=max(mst[i,j],mst[j,i])
        print(mst)
        graph=[]
        for i in range(data_size):

            k=np.argsort(dm[i])
            k=np.append(k[1:K+1],np.where(mst[i]>0))
            k=np.unique(k)
            graph.append(k)

            #k=np.where(mst[i] > 0)
            #graph.append(k)
        print(graph)
        return graph,mst


    def random_other_idx(sub_idxs):
        ret = random.randint(0, data_size - 1)
        if ret in sub_idxs:
            return random_other_idx(sub_idxs)
        return ret

    def swap_one(sub_idxs, swap_idx):
        sub_idxs = deepcopy(sub_idxs)
        sub_idxs[swap_idx] = random_other_idx(sub_idxs)
        return sub_idxs

    def swap_(sub_idxs, swap_idx,new_idx):
        sub_idxs = deepcopy(sub_idxs)
        sub_idxs[swap_idx] = new_idx
        return sub_idxs

    def del_one(sub_idxs, del_idx,st):
        sub_idxs = deepcopy(sub_idxs)
        x=np.array(range(st),dtype=int)
        #print(x)
        return sub_idxs[x!=int(del_idx)]

    def init():
        ans=np.array([],dtype=int)
        for i in range(n_samples*1000):
            topv=0
            topx=0
            topy=0
            for j in range(data_size):
                k=np.argmax(mst[j])

                if(mst[j,k]>topv):
                    topv=mst[j,k]
                    topx=j
                    topy=k
            ans=np.append(ans,topx)
            #ans=np.append(ans,topy)
            ans=np.unique(ans)
            mst[topx,topy]=0
            mst[topy,topx]=0
            if(ans.shape[0]==n_samples):
                break
        return ans




    def get_subset_score(X_sub, Y_sub):
        return KNNLoss(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)
        #return evalModel(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)

    def getscore(X_sub, Y_sub):

        return evalModel(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)

    #sub_idxs = list(range(n_samples))

    def shrinkinit(st):
        sub_idxs = np.array(range(st))
        X_sub, Y_sub = X[sub_idxs, :], Y[sub_idxs]
        while(st>n_samples):
            score_old=-1e8
            perma_id=deepcopy(sub_idxs)
            for i in range(st):
                new_sub_idxs = del_one(perma_id, i, st)
                new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]
                score_new = get_subset_score(new_X_sub, new_Y_sub)
                if score_new > score_old:
                    print("[knn_anneal] iteration ", i, " new score! : ", score_new)
                    X_sub, Y_sub = new_X_sub, new_Y_sub
                    sub_idxs = new_sub_idxs
                    score_old = score_new
                    #stuck_cnt = 0
                    sw = 1
            st=st-1
            print(st)
        return sub_idxs




    g,mst=MST()
    #sub_idxs = init()

    sub_idxs=shrinkinit(n_samples+1)
    print(sub_idxs)
    X_sub, Y_sub = X[sub_idxs, :], Y[sub_idxs]
    score_old = get_subset_score(X_sub, Y_sub)
    stuck_cnt = 0
    swap_idx = 0
    changed = 0
    for i in range(n_samples * 1000000):
        stuck_cnt += 1
        if(i%10==0):
            print("new score!",getscore(X_sub, Y_sub))
        if changed > 7 or i%1000 == 999:
            changed = 0
            for j in range(n_samples):
                #break
                # print(j)
                swap_idx = j
                score_old = get_subset_score(X_sub, Y_sub)
                while (True):
                    sw = 0
                    # node=sub_idxs[swap_idx]
                    for J in g[sub_idxs[swap_idx]]:
                        # print(j)
                        new_sub_idxs = swap_(sub_idxs, swap_idx, J)
                        new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]
                        score_new = get_subset_score(new_X_sub, new_Y_sub)
                        if score_new > score_old:
                            print("[knn_anneal] iteration ", i, "stuck ", 0, " new score! : ", score_new)
                            X_sub, Y_sub = new_X_sub, new_Y_sub
                            sub_idxs = new_sub_idxs
                            score_old = score_new
                            stuck_cnt = 0
                            sw = 1
                    if (sw == 0):
                        break
                    #break

        if stuck_cnt > n_samples*500:
            break
        swap_idx = i % n_samples
        maxscore=score_old

        score_old = getscore(X_sub, Y_sub)
        for j in range(1):
            #break
            #print(j)
            new_sub_idxs=swap_one(sub_idxs,swap_idx)
            new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]
            score_new = getscore(new_X_sub, new_Y_sub)
            if score_new > score_old:
                print("[knn_anneal] iteration ", i, "stuck ", stuck_cnt, " new score! : ", score_new)
                X_sub, Y_sub = new_X_sub, new_Y_sub
                sub_idxs = new_sub_idxs
                score_old = score_new
                stuck_cnt = 0
                changed = changed + 1
        #node=sub_idxs[swap_idx]


        if(stuck_cnt == -1):
            #swap_idx = j
            while (True):
                sw = 0
                # node=sub_idxs[swap_idx]
                for J in g[sub_idxs[swap_idx]]:
                    # print(j)
                    new_sub_idxs = swap_(sub_idxs, swap_idx, J)
                    new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]
                    score_new = get_subset_score(new_X_sub, new_Y_sub)
                    if score_new > score_old:
                        print("[knn_anneal] iteration ", i, "stuck ", stuck_cnt, " new score! : ", score_new)
                        X_sub, Y_sub = new_X_sub, new_Y_sub
                        sub_idxs = new_sub_idxs
                        score_old = score_new
                        stuck_cnt = 0
                        sw = 1
                if (sw == 0):
                    break




    return X[sub_idxs, :], Y[sub_idxs]

def sub_select_knn_MST_init_KNN_layer(X, Y, n_samples,layer = 5, big = False):
    from copy import deepcopy
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import distance_matrix
    data_size = len(X)

    def MST(X,K=5):
        dm=distance_matrix(X, X)
        print(dm)
        mst = minimum_spanning_tree(dm)
        mst=mst.toarray().astype(int)
        global data_size
        data_size = X.shape[0]
        for i in range(data_size):
            for j in range(data_size):
                mst[i,j]=max(mst[i,j],mst[j,i])
        print(mst)
        graph=[]
        for i in range(data_size):

            k=np.argsort(dm[i])
            k=np.append(k[1:K+1],np.where(mst[i]>0))
            k=np.unique(k)
            graph.append(k)

            #k=np.where(mst[i] > 0)
            #graph.append(k)
        print(graph)
        return graph,mst


    def random_other_idx(sub_idxs,k):
        ret = random.randint(0, k - 1)
        if ret in sub_idxs:
            return random_other_idx(sub_idxs,k)
        return ret

    def swap_one(sub_idxs, swap_idx,k):
        sub_idxs = deepcopy(sub_idxs)
        sub_idxs[swap_idx] = random_other_idx(sub_idxs,k)
        return sub_idxs

    def swap_(sub_idxs, swap_idx,new_idx):
        sub_idxs = deepcopy(sub_idxs)
        sub_idxs[swap_idx] = new_idx
        return sub_idxs

    def del_one(sub_idxs, del_idx,st):
        sub_idxs = deepcopy(sub_idxs)
        x=np.array(range(st),dtype=int)
        #print(x)
        return sub_idxs[x!=int(del_idx)]

    def init():
        ans=np.array([],dtype=int)
        for i in range(n_samples*1000):
            topv=0
            topx=0
            topy=0
            for j in range(data_size):
                k=np.argmax(mst[j])

                if(mst[j,k]>topv):
                    topv=mst[j,k]
                    topx=j
                    topy=k
            ans=np.append(ans,topx)
            #ans=np.append(ans,topy)
            ans=np.unique(ans)
            mst[topx,topy]=0
            mst[topy,topx]=0
            if(ans.shape[0]==n_samples):
                break
        return ans




    def get_subset_score(X_sub, Y_sub,Xgl=Xgl,Ygl=Ygl):
        #return KNNLoss(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)
        return evalModel(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)

    def getscore(X_sub, Y_sub,Xgl=Xgl,Ygl=Ygl):

        return evalModel(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)

    #sub_idxs = list(range(n_samples))



    def edge_walk(swap_idx,g,sub_idxs):
        while (True):
            sw = 0
            node=sub_idxs[swap_idx]
            for J in g[node]:
                # print(j)
                new_sub_idxs = swap_(sub_idxs, swap_idx, J)
                new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]
                score_new = get_subset_score(new_X_sub, new_Y_sub)
                if score_new > score_old:
                    print("[knn_anneal] iteration ", i, "stuck ", 0, " new score! : ", score_new)
                    X_sub, Y_sub = new_X_sub, new_Y_sub
                    sub_idxs = new_sub_idxs
                    score_old = score_new
                    stuck_cnt = 0
                    sw = 1
            if (sw == 0):
                break
        return sub_idxs

    def get_result(X,Y,sub_idxs):
        data_size = X.shape[0]
        g, mst = MST(X)

        print(sub_idxs)

        X_sub, Y_sub = X[sub_idxs, :], Y[sub_idxs]
        score_old = get_subset_score(X_sub, Y_sub)
        stuck_cnt = 0
        swap_idx = 0
        changed = 0
        for i in range(1):
            stuck_cnt += 1
            if (i % 1 == 0):
                print("new score!", getscore(X_sub, Y_sub))
            if changed > 7 or i % 1 == 0:
                changed = 0
                while(True):
                    sw=0
                    for j in range(n_samples):
                        # break
                        # print(j)
                        swap_idx = j
                        score_old = get_subset_score(X_sub, Y_sub,X,Y)

                        # node=sub_idxs[swap_idx]
                        for J in g[sub_idxs[swap_idx]]:
                            # print(j)
                            new_sub_idxs = swap_(sub_idxs, swap_idx, J)
                            new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]
                            score_new = get_subset_score(new_X_sub, new_Y_sub,X,Y)
                            if score_new > score_old:
                                print("[knn_anneal] iteration ", i, "stuck ", 0, " new score! : ", score_new)
                                X_sub, Y_sub = new_X_sub, new_Y_sub
                                sub_idxs = new_sub_idxs
                                score_old = score_new
                                stuck_cnt = 0
                                sw = 1
                                break
                    if (sw == 0):
                        break

                            # break


            if stuck_cnt > n_samples * 500:
                break
            swap_idx = i % n_samples
            maxscore = score_old

            score_old = getscore(X_sub, Y_sub,X,Y)
            for j in range(1):
                break
                # print(j)
                new_sub_idxs = swap_one(sub_idxs, swap_idx,data_size)
                new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]
                score_new = getscore(new_X_sub, new_Y_sub,X,Y)
                if score_new > score_old:
                    print("[knn_anneal] iteration ", i, "stuck ", stuck_cnt, " new score! : ", score_new)
                    X_sub, Y_sub = new_X_sub, new_Y_sub
                    sub_idxs = new_sub_idxs
                    score_old = score_new
                    stuck_cnt = 0
                    changed = changed + 1
            # node=sub_idxs[swap_idx]

            if (stuck_cnt == -1):
                # swap_idx = j
                while (True):
                    sw = 0
                    # node=sub_idxs[swap_idx]
                    for J in g[sub_idxs[swap_idx]]:
                        # print(j)
                        new_sub_idxs = swap_(sub_idxs, swap_idx, J)
                        new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]
                        score_new = get_subset_score(new_X_sub, new_Y_sub)
                        if score_new > score_old:
                            print("[knn_anneal] iteration ", i, "stuck ", stuck_cnt, " new score! : ", score_new)
                            X_sub, Y_sub = new_X_sub, new_Y_sub
                            sub_idxs = new_sub_idxs
                            score_old = score_new
                            stuck_cnt = 0
                            sw = 1
                    if (sw == 0):
                        break

        return sub_idxs






    sub_idxs = list(range(n_samples))
    num=X.shape[0]
    for i in range(layer):
        num=int(num/2)



    for i in range(layer+1):
        #getmore(num)
        sub_idxs = get_result(X[0:num,:],Y[0:num],sub_idxs)

        num = num*2
        if(i==layer):
            num = X.shape[0]

    return X[sub_idxs,:],Y[sub_idxs]


def sub_select_knn_swap(X, Y, n_samples):
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

    def swap_given(sub_idxs, swap_idx, given):
        sub_idxs = deepcopy(sub_idxs)
        sub_idxs[swap_idx] = given
        return sub_idxs


    def get_subset_score(X_sub, Y_sub):
        #return KNNLoss(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)
        return evalModel(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)

    def getscore(X_sub, Y_sub):
        #return KNNLoss(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)
        return evalModel(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)


    sub_idxs = list(range(n_samples))
    X_sub, Y_sub = X[sub_idxs, :], Y[sub_idxs]

    score_old = get_subset_score(X_sub, Y_sub)
    stuck_cnt = 0
    for i in range(n_samples * 10000000):
        stuck_cnt += 1
        if (i % 10 == 0):
            print("new score!", getscore(X_sub, Y_sub))
        if stuck_cnt > n_samples*50:
            break
        swap_idx = i % n_samples
        given = random_other_idx(sub_idxs)

        new_sub_idxs = swap_given(sub_idxs, swap_idx, given)
        new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]

        score_new = get_subset_score(new_X_sub, new_Y_sub)
        maxscore = get_subset_score(X_sub, Y_sub)
        swp = 0

        if score_new > score_old:

            for j in range(n_samples):
                new_sub_idxs = swap_given(sub_idxs, j, given)
                new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]
                score_new = get_subset_score(new_X_sub, new_Y_sub)
                if score_new > maxscore:
                    maxscore = score_new
                    swp = j

            new_sub_idxs = swap_given(sub_idxs, swp, given)
            new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]
            score_new = get_subset_score(new_X_sub, new_Y_sub)

            print("[knn_anneal] iteration ", i, "stuck ", stuck_cnt, " new score! : ", score_new)
            X_sub, Y_sub = new_X_sub, new_Y_sub
            sub_idxs = new_sub_idxs
            score_old = score_new
            stuck_cnt = 0

    return X[sub_idxs, :], Y[sub_idxs]

class KNN_opt:
    from collections import deque

    class captain:
        def __init__(self):
            self.crew = []
            self.radius = 0.0




    def __init__(self,K,num_captain,n):
        self.name="KNN"
        self.K=K
        self.ini = 0
        self.num_captain = num_captain
        self.captains = [self.captain() for i in range(num_captain) ]


        self.cor = 0
        self.corr = []
        self.cache = np.zeros(n)

    def set_crew(self,X,Y):
        self.X = X
        self.Y = Y
        self.corr=np.zeros(self.X.shape[0])

    def assign_captain(self,i,X,Y):
        dis = np.zeros(X.shape[0])
        for j in range(X.shape[0]):
            dis[j] = self.dist(self.X[i], X[j])
        k = np.argsort(dis)
        ans = np.zeros(2)
        for j in range(self.K):
            ans[Y[k[j]]] = ans[Y[k[j]]] + 1
        self.captains[k[0]].crew.append(i)
        #print(i,k[0])
        #if(i<100):
        #    print(self.captains[k[0]].crew)
        self.captains[k[0]].radius = max(self.captains[k[0]].radius,dis[k[0]]+dis[k[self.K-1]])
        return np.argmax(ans)

    def predict_captain(self,i,X,Y):
        dis = np.zeros(X.shape[0])
        for j in range(X.shape[0]):
            dis[j] = self.dist(self.X[i], X[j])
        k = np.argsort(dis)
        ans = np.zeros(2)
        for j in range(self.K):
            ans[Y[k[j]]] = ans[Y[k[j]]] + 1
        return np.argmax(ans)

    def swap(self,X,Y,id=-1,x=np.array([]),y=-1): #after swaping
        if(id == -1):
            for i in range(self.X.shape[0]):
                #print(i)
                predicted = self.assign_captain(i,X,Y)
                if(predicted == self.Y[i]):
                    self.cor = self.cor + 1
                    self.corr[i]=1

        else:
            tt = 0
            for i in range(self.num_captain):

                if(i == id or self.dist(x,X[i])<self.captains[i].radius or self.dist(X[id],X[i])<self.captains[i].radius):

                    self.captains[i].radius = 0
                    for j in self.captains[i].crew:
                        self.cache[tt] = j
                        tt = tt+1
                    self.captains[i].crew = []
            for i in range(tt):
                j = int(self.cache[i])
                self.cor = self.cor - self.corr[j]
                self.corr[j] = 0
                predicted = self.assign_captain(j, X,Y)
                if (predicted == self.Y[j]):
                    self.cor = self.cor + 1
                    self.corr[j] = 1



        return self.cor/self.X.shape[0]

    def predict(self, X, Y, id=-1, x=np.array([]), y=-1):  # after swaping

        cor = self.cor
        tt = 0
        for i in range(self.num_captain):
            #print(i,cor)
            if (i==id or self.dist(x, X[i]) < self.captains[i].radius or self.dist(X[id], X[i]) < self.captains[i].radius):
                tt = tt+1
                for j in self.captains[i].crew:
                    cor = cor - self.corr[j]
                    predicted = self.predict_captain(j, X,Y)
                    if (predicted == self.Y[j]):
                        cor = cor + 1
        #print(tt)
        return cor/self.X.shape[0]



    def dist(self,x1,x2):
        return np.linalg.norm(x2-x1)


class KNN_opt_1:
    from collections import deque

    class captain:
        def __init__(self):
            self.crew = []
            self.radius = 0.0




    def __init__(self,K,num_captain,n):
        self.name="KNN"
        self.K=K
        self.ini = 0
        self.num_captain = num_captain
        self.captains = [self.captain() for i in range(num_captain) ]


        self.cor = 0
        self.corr = []
        self.cache = np.zeros(n)

    def set_crew(self,X,Y):
        self.X = X
        self.Y = Y
        self.corr=np.zeros(self.X.shape[0])

    def assign_captain(self,i,X,Y):
        dis = np.zeros(X.shape[0])
        for j in range(X.shape[0]):
            dis[j] = self.dist(self.X[i], X[j])
        k = np.argsort(dis)
        ans = np.zeros(2)
        for j in range(self.K):
            ans[Y[k[j]]] = ans[Y[k[j]]] + 1
        self.captains[k[0]].crew.append(i)
        #print(i,k[0])
        #if(i<100):
        #    print(self.captains[k[0]].crew)
        self.captains[k[0]].radius = max(self.captains[k[0]].radius,dis[k[0]]+dis[k[self.K-1]])
        return np.argmax(ans)

    def predict_captain(self,i,X,Y):
        dis = np.zeros(X.shape[0])
        for j in range(X.shape[0]):
            dis[j] = self.dist(self.X[i], X[j])
        k = np.argsort(dis)
        ans = np.zeros(2)
        for j in range(self.K):
            ans[Y[k[j]]] = ans[Y[k[j]]] + 1
        return np.argmax(ans)

    def swap(self,X,Y,id=-1,x=np.array([]),y=-1): #after swaping
        if(id == -1):
            for i in range(self.X.shape[0]):
                #print(i)
                predicted = self.assign_captain(i,X,Y)
                if(predicted == self.Y[i]):
                    self.cor = self.cor + 1
                    self.corr[i]=1

        else:
            tt = 0
            for i in range(self.num_captain):

                if(i == id or self.dist(x,X[i])<self.captains[i].radius or self.dist(X[id],X[i])<self.captains[i].radius):

                    self.captains[i].radius = 0
                    for j in self.captains[i].crew:
                        self.cache[tt] = j
                        tt = tt+1
                    self.captains[i].crew = []
            for i in range(tt):
                j = int(self.cache[i])
                self.cor = self.cor - self.corr[j]
                self.corr[j] = 0
                predicted = self.assign_captain(j, X,Y)
                if (predicted == self.Y[j]):
                    self.cor = self.cor + 1
                    self.corr[j] = 1



        return self.cor/self.X.shape[0]

    def predict(self, X, Y, id=-1, x=np.array([]), y=-1):  # after swaping

        cor = self.cor
        for i in range(self.num_captain):
            #print(i,cor)
            if (i==id or self.dist(x, X[i]) < self.captains[i].radius or self.dist(X[id], X[i]) < self.captains[i].radius):
                for j in self.captains[i].crew:
                    cor = cor - self.corr[j]
                    predicted = self.predict_captain(j, X,Y)
                    if (predicted == self.Y[j]):
                        cor = cor + 1
        return cor/self.X.shape[0]



def sub_select_knn_opt(X, Y, n_samples):
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
        #return KNNLoss(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)
        return evalModel(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)

    def getscore(X_sub, Y_sub):
        #return KNNLoss(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)
        return evalModel(X_sub, Y_sub, Xgl, Ygl, k=KNNpoint)


    KNN = KNN_opt(KNNpoint,n_samples,X.shape[0])
    KNN.set_crew(Xgl,Ygl)
    sub_idxs = list(range(n_samples))
    X_sub, Y_sub = X[sub_idxs, :], Y[sub_idxs]

    score_old = KNN.swap(X_sub,Y_sub)
    #print(score_old)
    stuck_cnt = 0
    for i in range(n_samples * 10000000):
        stuck_cnt += 1
        if (i % 10 == 0):
            print("new score!", getscore(X_sub, Y_sub))
        if stuck_cnt > n_samples:
            break
        swap_idx = i % n_samples
        new_sub_idxs = swap_one(sub_idxs, swap_idx)
        new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]
        #print("zzz")
        score_new = KNN.predict(new_X_sub, new_Y_sub,swap_idx,X_sub[swap_idx,Y_sub[swap_idx]])
        #print(score_new,getscore(new_X_sub,new_Y_sub))

        if score_new > score_old:
            print("[knn_anneal] iteration ", i, "stuck ", stuck_cnt, " new score! : ", score_new)
            X_sub, Y_sub = new_X_sub, new_Y_sub
            sub_idxs = new_sub_idxs
            score_old = KNN.swap(new_X_sub, new_Y_sub,swap_idx,X_sub[swap_idx,Y_sub[swap_idx]])
            stuck_cnt = 0

    return X[sub_idxs, :], Y[sub_idxs]


def KNNSVM(k=10):
    knn=KNN(int(dat/k)*KNNpoint)
    subX,subY=knn.goodpoints()
    svm = SVMSKT()

    #print(X.shape, Y.shape)
    svm.fit(subX, subY)
    #print(svm.alpha)
    #print(svm.sv())
    print(svm.test(X, Y))
    #together=svm.top(k)
    #for zz in together:
    #    print (zz[1], zz[0])
    return svm.top(k,False)



def KNNCCL(k=10):
    knn = KNN(int(dat / k) * KNNpoint)
    subX, subY = knn.goodpoints()
    ccl = Circle()
    ccl.learn((subX, subY))
    #a,b=zip(*together)
    #print(a,b)
    return ccl.top(k,False)

def KNNSSL(k=10):
    knn = KNN(int(dat / k) * KNNpoint)
    subX, subY = knn.goodpoints()
    return sub_select_knn(subX,subY,chosepoint)

def luangao(leng,t,ratio,chosepoint):
#    print(X[1:10])
    global Xgl,Ygl
    global X,Y
    rchosepoint = chosepoint
    from copy import deepcopy
    #leng = X.shape[0] / subgroup


    for i in range(t):
        subgroup = int(X.shape[0]/leng)
        #leng = X.shape[0]/subgroup
        ansX=np.array([])
        ansY = np.array([])
        for j in range(subgroup):
            print(i,j)
            st=int(j*leng)
            ed=int(st+leng)-1
            Xgl=deepcopy(X[st:ed+1])
            Ygl=deepcopy(Y[st:ed+1])
            chosepoint = int(leng*ratio)
            subX,subY=sub_select_knn(X[st:(ed+1)],Y[st:(ed+1)],chosepoint)
            if(j==0):
                ansX=subX
            else:
                ansX=np.append(ansX,subX,axis=0)
            ansY=np.append(ansY,subY)
        X=deepcopy(ansX)
        Y=deepcopy(ansY)
        Y=Y.astype(int)

    print("let the games begin",Xgl.shape,Ygl.shape)


    Xgl,Ygl = X,Y
    #Xgl,Ygl = X[1:700,:],Y[1:700]

    print('Let the games begin',Xgl.shape,Ygl.shape)
    return sub_select_knn_opt(X,Y,rchosepoint)







if __name__ == '__main__':
  dat=2000
  global X,Y
  X,Y = make_dataset(dat)
  XX=deepcopy(X)
  YY=deepcopy(Y)
  Xgl,Ygl=X,Y
  print(X.shape,Y.shape)
  together = zip(X,Y)
  chosepoint=50
  KNNpoint=1
  #xtp,ytp=KNNCCL(chosepoint)
  #xtp1,ytp1=CCL(chosepoint)
  starttime = time.time()
  #xt,yt=sub_select_knn(X, Y, chosepoint)

  #xmst,ymst=sub_select_knn_MST_init_KNN_layer(X, Y, chosepoint)

  xt1,yt1=luangao(100,2,0.25,chosepoint)
  print(time.time()-starttime)
  #print("our",evalModel(xmst,ymst,X,Y,KNNpoint))
  #print("our origininal", evalModel(xtp1, ytp1, X, Y, KNNpoint))
  #print("to beat",evalModel(xt,yt,XX,YY,KNNpoint))
  print("to beat updated", evalModel(xt1, yt1, XX, YY, KNNpoint))
  print("random",RNDtest(chosepoint,KNNpoint))
  #print(np.argsort(np.array(1:5)))

  #for zz in together:
  #  print (zz[1], zz[0])

  #500 points choose 50, K=5
  #SVM, max alpha, Gaussian: 0.682
  #SVM, min alpha, Gaussian: 0.65
  #LGR: 0.522
  #CirCle: 0.686
  #Random: 0.666

#Choose 10, K=2:
#Random: 0.624
#Circle: 0.608

#50,2
#RND:0.686
#Circle: 0.676
#SVM: max alpha: 0.686
#SVM: min alpha: 0.638
#SVM, linear kernel min alpha: 0.478
#SVM, linear, max alpha: 0.532
#SVM,

#50, 1
#RND, 0.7
#Circle, 0.716
#SVM: max alpha: 0.718
#SVM: min alpha: 0.676

#%60 for KNNSVM

#439 0.792

# to beat 787 in 1 min
# 82.3forever

#75.8
# 79

# 84.1

#78 75


#raw: 177s, 0.771

#5,2,0.2,230s,0.697
#5,2,0.5,2275s,0.753

#20,4,0.5,10s,0.6655
#100,4,0.5,170s,0.7335

#80,2,0.25,27s,0.67
#40,7,0.75,92s,0.71
#50,4,0.5,45s,0.70
#100,2,0.25,41s,0.70
