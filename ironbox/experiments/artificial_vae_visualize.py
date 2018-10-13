
if __name__ == "__main__":
    from datas.artificial_classify import gen_data
    from evaluation_models.data_holder import DataHolder
    X_tr, Y_tr, X_t, Y_t = gen_data(2000)

    from auto_encoders.fc_bit_vae import FcVAE
    vae = FcVAE(20, 2)

    saved_model_path = 'saved_models/artificial_vae.mdl'
    import os.path
    if os.path.isfile(saved_model_path):
        print ("loaded saved model at ", saved_model_path)
        vae.load(saved_model_path)
    else:
        print ("no saved model found, training auto-encoder ")
        vae.learn(X_tr)
        vae.save(saved_model_path)

    # compute the embedded features in 2D
    X_tr_emb = vae.embed(X_tr)
    print (X_tr_emb.shape)

    # get the prototype subset
    from subset_selection.subset_selection import sub_select_knn
    n_sub = 100
    sub_idxs = sub_select_knn(X_tr, Y_tr, n_sub, vae)

    from sklearn.metrics import pairwise_distances_argmin_min
    
    # get the cluster assignments for each point in embedded space
    X_centers = X_tr_emb[sub_idxs, :2]
    closest, _ = pairwise_distances_argmin_min(X_tr_emb, X_centers)


    # begin the plotting
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(dpi=2400)

    # plot the cluster assignments (lines)
    from scipy.spatial import Voronoi, voronoi_plot_2d
    vor = Voronoi(X_centers)
    voronoi_plot_2d(vor, show_vertices=False)

    # plot the tsne embedding space
    cl_colors = np.linspace(0, 1, len(set(Y_tr)))
    # matplotlib.use("svg")
    x1 = [x[0] for x in X_tr_emb]
    x2 = [x[1] for x in X_tr_emb]
    colors = [cl_colors[lab] for lab in Y_tr]
    plt.scatter(x1, x2, c=colors, alpha=0.5, edgecolors='none')


    plt.savefig('artificial_embed.png')
    plt.clf()




