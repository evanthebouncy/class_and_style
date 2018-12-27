def embed_artificial():
    emb_dim = 2 

    import pickle
    X_tr, Y_tr, X_t, Y_t = pickle.load(open('data_raw/artificial/artificial.p', 'rb'))

    from embed.fc_vae import FcVAE
    vae = FcVAE(20, emb_dim, 'xentropy')

    saved_model_path = 'embed/saved_models/artificial_vae_{}.mdl'.format(emb_dim)
    import os.path
    if os.path.isfile(saved_model_path):
        print ("loaded saved model at ", saved_model_path)
        vae.load(saved_model_path)
    else:
        print ("no saved model found, training auto-encoder ")
        vae.learn(X_tr, learn_iter = 20000)
        vae.save(saved_model_path)
        print ("saved model at ", saved_model_path)

    # compute the embedded features
    X_tr_emb = vae.embed(X_tr)

    data_embed_path = 'data_embed/artificial_dim{}.p'.format(emb_dim)
    pickle.dump((X_tr_emb,Y_tr), open( data_embed_path, "wb" ) )

    data_embed_path = 'data_embed/artificial_noemb.p'
    pickle.dump((X_tr, Y_tr), open( data_embed_path, "wb" ) )

def embed_mnist():
    emb_dim = 32
    from data_raw.mnist_ import gen_data
    X_tr, Y_tr, X_t, Y_t = gen_data("./data_raw")

    import pickle

    X_size = X_tr.shape[1]
    from embed.fc_vae import FcVAE
    vae = FcVAE(X_size, emb_dim, 'L2')

    saved_model_path = 'embed/saved_models/mnist_vae_{}.mdl'.format(emb_dim)
    import os.path
    if os.path.isfile(saved_model_path):
        print ("loaded saved model at ", saved_model_path)
        vae.load(saved_model_path)
    else:
        print ("no saved model found, training auto-encoder ")
        vae.learn(X_tr, learn_iter = 100000)
        vae.save(saved_model_path)
        print ("saved model at ", saved_model_path)

    # compute the embedded features
    X_tr_emb = vae.embed(X_tr)

    data_embed_path = 'data_embed/mnist_dim{}.p'.format(emb_dim)
    pickle.dump((X_tr_emb,Y_tr), open( data_embed_path, "wb" ) )


if __name__ == "__main__":
    # embed_artificial()
    embed_mnist()

