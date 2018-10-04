
if __name__ == "__main__":
  from datas.artificial_classify import gen_data
  from evaluation_models.data_holder import DataHolder
  X_tr, Y_tr, X_t, Y_t = gen_data(2000)

  from auto_encoders.fc_bit_auto_encoder import FcAE
  ae = FcAE(20, 10)

  saved_model_path = 'saved_models/artificial_ae.mdl'
  import os.path
  if os.path.isfile(saved_model_path):
    print ("loaded saved model at ", saved_model_path)
    ae.load(saved_model_path)
  else:
    print ("no saved model found, training auto-encoder ")
    ae.learn(X_tr)
    ae.save(saved_model_path)

  # compute the embedded features in 2D
  X_tr_emb = ae.embed(X_tr)
  print (X_tr_emb.shape)

  # plot the embedding space
  import matplotlib.pyplot as plt
  import numpy as np
  cl_colors = np.linspace(0, 1, len(set(Y_tr)))
  # matplotlib.use("svg")
  x = [x[0] for x in X_tr_emb]
  y = [x[1] for x in X_tr_emb]
  colors = [cl_colors[lab] for lab in Y_tr]
  plt.scatter(x, y, c=colors, alpha=0.5)
  plt.savefig('artificial_embed.png')
  plt.clf()

