import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
from visualizer import getdata
from cnn_vae import to_torch


def embed_Relation():
    emb_dim = 32
    images, questions, answers = getdata()
    N = len(answers)
    X_tr = images / 255.0

    Y_tr = np.array(answers).astype(int)
    _, Y_tr = (np.where(Y_tr != 0))

    import pickle
    
    X_img_tr = np.array(X_tr)
    X_qry_tr = np.array(questions).astype(float)

    from cnn_vae import ECLVR
    cnn = ECLVR(3).cuda()

    saved_model_path = 'relation_model.mdl'
    import os.path

    retrain = True

    if os.path.isfile(saved_model_path):
        print ("loaded saved model at ", saved_model_path)
        cnn.load(saved_model_path)
    else:
        print ("no saved model found, training auto-encoder ")
        cnn.learn(X_img_tr, X_qry_tr, learn_iter = 50000)
        cnn.save(saved_model_path)
        print ("saved model at ", saved_model_path)

    # compute the embedded features
    amt = 1000
    X_embs = []
    for i in range(N // amt):
        X_tr_emb = cnn.embed(to_torch(X_img_tr[i * amt : (i+1) * amt]), 
                             to_torch(X_qry_tr[i * amt : (i+1) * amt]) )
        X_tr_emb = X_tr_emb.detach().cpu().numpy()
        X_embs.append(X_tr_emb)

    X_embs = np.concatenate(X_embs, axis=0)
    print (X_embs.shape)
    data_embed_path = 'relation_dim{}.p'.format(emb_dim)
    pickle.dump((X_embs,Y_tr), open( data_embed_path, "wb" ) )
    # data_embed_path = 'relation_dec.p'
    # pickle.dump((X_tr_dec), open( data_embed_path, "wb" ) )

if __name__ =='__main__':
    embed_Relation()
