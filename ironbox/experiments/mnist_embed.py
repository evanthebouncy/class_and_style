# ======================= HELPERS =====================

if __name__ == "__main__":
    from datas.mnist_classify import gen_data
    from evaluation_models.data_holder import DataHolder
    X_tr, Y_tr, X_t, Y_t = gen_data('./datas')


    from auto_encoders.cnn_auto_encoder import CnnAE
    ae = CnnAE(1, 28)

    saved_model_path = 'saved_models/mnist_ae.mdl'
    import os.path
    if os.path.isfile(saved_model_path):
        print ("loaded saved model at ", saved_model_path)
        ae.load(saved_model_path)
    else:
        print ("no saved model found, training auto-encoder ")
        ae.learn(X_tr)
        ae.save(saved_model_path)


    X_tr_emb = ae.embed(X_tr)
    import pickle
    save_dir = './saved_embeddings/' + 'mnist_dim32.p'
    pickle.dump(X_tr_emb, open( save_dir, "wb" ) )
    print ('embedding saved at ', save_dir)
