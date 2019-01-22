import numpy as np

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

def embed_fashion():
    emb_dim = 32
    from data_raw.fashion_mnist_ import gen_data
    X_tr, Y_tr, X_t, Y_t = gen_data("./data_raw")

    import pickle

    X_size = X_tr.shape[1]
    from embed.fc_vae import FcVAE
    vae = FcVAE(X_size, emb_dim, 'L2')

    saved_model_path = 'embed/saved_models/fashion_vae_{}.mdl'.format(emb_dim)
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

    data_embed_path = 'data_embed/fashion_dim{}.p'.format(emb_dim)
    pickle.dump((X_tr_emb,Y_tr), open( data_embed_path, "wb" ) )

def embed_mnist_usps():
    import pickle
    emb_dim = 32
    from data_raw.mnist_ import gen_data as mnist_gen_data
    MNIST_X_tr, MNIST_Y_tr, MNIST_X_t, MNIST_Y_t = mnist_gen_data("./data_raw")
    from data_raw.usps import gen_data as usps_gen_data
    USPS_X_tr, USPS_Y_tr, USPS_X_t, USPS_Y_t = usps_gen_data("./data_raw")

    X_tr = np.concatenate((MNIST_X_tr, USPS_X_tr), axis=0)
    Y_tr = np.concatenate((MNIST_Y_tr, USPS_Y_tr), axis=0)

    X_size = X_tr.shape[1]
    from embed.fc_vae import FcVAE
    vae = FcVAE(X_size, emb_dim, 'L2')

    saved_model_path = 'embed/saved_models/mnist_usps_vae_{}.mdl'.format(emb_dim)
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

    data_embed_path = 'data_embed/mnist_usps_dim{}.p'.format(emb_dim)
    pickle.dump((X_tr_emb,Y_tr), open( data_embed_path, "wb" ) )

def embed_mnist_skew():
    emb_dim = 32
    from data_raw.mnist_ import gen_data_skew
    X_tr, Y_tr, X_t, Y_t = gen_data_skew("./data_raw")

    import pickle

    X_size = X_tr.shape[1]
    from embed.fc_vae import FcVAE
    vae = FcVAE(X_size, emb_dim, 'L2')

    saved_model_path = 'embed/saved_models/mnist_skew_vae_{}.mdl'.format(emb_dim)
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

    data_embed_path = 'data_embed/mnist_skew_dim{}.p'.format(emb_dim)
    pickle.dump((X_tr_emb,Y_tr), open( data_embed_path, "wb" ) )

def embed_mdsd_dvd():
    emb_dim = 32
    from data_raw.mdsd_dvd import gen_data as dvd_gen_data
    X_tr, Y_tr, X_t, Y_t = dvd_gen_data("./data_raw")

    import pickle

    X_size = X_tr.shape[1]
    from embed.fc_vae import FcVAE
    vae = FcVAE(X_size, emb_dim, 'L2')

    saved_model_path = 'embed/saved_models/mdsd_dvd_vae_{}.mdl'.format(emb_dim)
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

    data_embed_path = 'data_embed/mdsd_dvd_dim{}.p'.format(emb_dim)
    pickle.dump((X_tr_emb,Y_tr), open( data_embed_path, "wb" ) )

def embed_sentiment():
    emb_dim = 32
    from data_raw.sentiment_ import gen_data
    X_tr, Y_tr, X_t, Y_t = gen_data("./data_raw")

    import pickle

    X_size = X_tr.shape[1]
    from embed.fc_vae import FcVAE
    vae = FcVAE(X_size, emb_dim, 'L2', output_type='linear')

    saved_model_path = 'embed/saved_models/sentiment_vae_{}.mdl'.format(emb_dim)
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

    data_embed_path = 'data_embed/sentiment_dim{}.p'.format(emb_dim)
    pickle.dump((X_tr_emb,Y_tr), open( data_embed_path, "wb" ) )

    orig_embed_path = 'data_embed/sentiment_dim_bert.p'
    pickle.dump((X_tr,Y_tr), open( orig_embed_path, "wb" ) )


if __name__ == "__main__":
    # embed_artificial()
    # embed_mnist()
    # embed_fashion()
    # embed_mnist_usps()
    # embed_mnist_skew()
    #embed_mdsd_dvd()
    embed_sentiment()

