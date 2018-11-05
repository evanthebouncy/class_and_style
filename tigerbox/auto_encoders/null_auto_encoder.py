class NullAE():
  def __init__(self):
    print ("hi i am null autoencoder")

  def learn(self, X_train):
    print ("learning ae by not doing anything")
    return

  def embed(self, X_train):
    return X_train
