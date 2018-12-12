import numpy as np
from sklearn import neighbors

def score_subset(X_sub, Y_sub, X, Y, W, one_near=False):
    K = 1 if one_near else 1+int(np.log(len(Y_sub)))
    clf = neighbors.KNeighborsClassifier(K)
    knn_cl = clf.fit(X_sub, Y_sub)
    Y_pred = knn_cl.predict(X)
    true_pred = (Y_pred == Y)
    score = true_pred * W - (1 - true_pred) * W
    return np.sum(score)

def update_weight(X_sub, Y_sub, X, Y, W, one_near=False):
    K = 1 if one_near else 1+int(np.log(len(Y_sub)))
    clf = neighbors.KNeighborsClassifier(K)
    knn_cl = clf.fit(X_sub, Y_sub)
    W_sub = np.zeros(Y_sub.shape)
    _, X_elders_idx = knn_cl.kneighbors(X)
    #print (X_elders_idx)
    X_elders_lab = np.take(Y_sub, X_elders_idx)
    #print (X_elders_lab)
    X_lab_expand = np.stack([Y for _ in range(K)]).transpose()
    #print (X_lab_expand)
    W_expand = np.stack([W for _ in range(K)]).transpose()
    #print ('W_expand\n', W_expand)
    elders_correct = X_elders_lab == X_lab_expand
    #print (elders_correct)
    elder_update = (1 * (elders_correct) + (-1) * (1 - elders_correct))
    elder_update = elder_update / K * W_expand
    #print (elder_update)
    #print (X_elders_idx)
    # TODO : maybe make this sparse matrix and not forloop LMAO but not now
    for e_idxs, e_ups in zip(X_elders_idx, elder_update):
        for idx, up in zip(e_idxs, e_ups):
            W_sub[idx] += up
    return W_sub


if __name__ == "__main__":
    # TEST 1
    def test1():
        X_sub = np.array([ [1,0,1],
                           [0,1,0] ])
        Y_sub = np.array([ 2, 3 ])
        X = np.array([ [1,0,1],
                       [0,1,0],
                       [1,1,1],
                       [1,0,0]])
        Y = np.array([2, 3, 1, 2])
        W = np.array([2, 3, 0, 1])
        score = score_subset(X_sub, Y_sub, X, Y, W)
        print (score)

    # TEST 2
    def test2():
        X_sub = np.array([ [1,0,1],
                           [0,1,0] ])
        Y_sub = np.array([ 2, 3 ])
        X = np.array([ [1,0,1],
                       [0,1,0],
                       [1,1,1],
                       [1,0,0]])
        Y = np.array([2, 3, 1, 2])
        W = np.array([2, 3, 0, 1])
        W_sub = update_weight(X_sub, Y_sub, X, Y, W)
        print (W_sub)

    test2()


# class KNN_W:
#     def __init__(self,K):
#         self.name = "KNN"
#         self.K = K
# 
#     def getRef(self,X,Y):
#         self.X = X
#         self.Y = Y
#         self.wei = np.zeros(Y.shape)
#         self.n_class = len(np.unique(Y))
# 
#     def dist(self,x1,x2):
#         return np.linalg.norm(x2-x1)
# 
#     # predict the y value given a x value
#     # has nothing to do with weight
#     def predict(self,x):
#         dis = np.zeros(self.X.shape[0])
#         for i in range(self.X.shape[0]):
#             dis[i] = self.dist(x,self.X[i])
#         k = np.argsort(dis)
#         ans = np.zeros(10)
#         for i in range(self.K):
#                 ans[self.Y[k[i]]] = ans[self.Y[k[i]]]+1
#         return np.argmax(ans)
# 
#     def assignwei(self,x,y,wei):
#         dis = np.zeros(self.X.shape[0])
#         for i in range(self.X.shape[0]):
#             dis[i] = self.dist(x,self.X[i])
#         k = np.argsort(dis)
#         if(self.Y[k[0]] =  = y):
#             self.wei[k[0]] = self.wei[k[0]]+wei
#         else:
#             self.wei[k[0]]  =  self.wei[k[0]] - wei
# 
# 
# 
# 
#     def eval(self,testX,testY,wei):
#         if(self.Y.shape[0] =  = 0):
#             return 0.0
#         ans = 0.0
#         for i in range(testX.shape[0]):
#             #print("have",i)
#             #print(self.dist(X[437],X[499]))
#             if(self.predict(testX[i]) =  = testY[i]):
#                 ans = ans+wei[i]
#             #print(self.predict(testX[i]),testY[i])
#         return ans/float(testX.shape[0])
# 
#     def getwei(self,testX,testY,wei):
#         if (self.Y.shape[0]  =  =  0):
#             return []
#         ans  =  0.0
#         for i in range(testX.shape[0]):
#             # print("have",i)
#             # print(self.dist(X[437],X[499]))
#             self.assignwei(testX[i],testY[i],wei[i])
#             # print(self.predict(testX[i]),testY[i])
#         return self.wei
# 
#     def goodpoints(self):
#         self.getRef(X,Y)
#         good = np.zeros(X.shape[0])
#         for i in range(X.shape[0]):
#             if(self.predict(X[i]) =  = Y[i]):
#                 good[i] = 1
#         return (X[good =  = 1],Y[good =  = 1])
# 
#     def loss(self,testX,testY): #larger the better
#         #return self.eval(testX,testY)
#         ans  =  0.0
#         for i in range(testX.shape[0]):
#             # print("have",i)
#             # print(self.dist(X[437],X[499]))
#             #if (self.predict(testX[i])  =  =  testY[i]):
#             #    ans  =  ans + 1
# 
#             #ans = ans+math.log(self.count(testX[i],testY[i]))
#             ans  =  ans + (self.count(testX[i], testY[i]))
#             # print(self.predict(testX[i]),testY[i])
#         return ans
# 
