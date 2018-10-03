# Subset Selection

Perform clustering on the latent space induced by ae (note ae here can be null so no embedding)

Takes in X, Y, n\_samples, and an autoencoder

Returns the INDEX of the subsets of X and Y (I return the index because in RL this become useful to use)

## sub\_select\_cluster(X, Y, n\_samples, ae)

## sub\_select\_cluster\_label(X, Y, n\_samples, ae)

## sub\_select\_knn\_anneal(X, Y, n\_samples, ae)

## sub\_select\_random(X, Y, n\_samples, ae)

