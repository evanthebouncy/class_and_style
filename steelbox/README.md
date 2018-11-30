# PROTOTYPE SUBSET SELECTION FOR STUFFS

## Stored Information

### data\_raw
stores the raw data, each data has a specific .py file with the function get\_data() which returns 4 outputs: Xtrain, Ytrain, Xtest, Ytest

### data\_embed
stores the embedded Xtrain, as a pickle of nparray

    dataname_embedname.p

### data\_sub
stores the index and weights of the selected subsets, stored as

    dataname_subsetsize_embedname_subsetalgorithm_meta.p

### results
stores the results, as

    dataname_subsetsize_embedname_subsetalgorithm_meta.p

## Code and Scripts

### embed
scripts and code for embedding, takes things from data\_raw and output things into data\_embed

### subset\_selection
scripts and code for producing subsets from the embeddings, takes things from data\_embed and output things into data\_sub

### evaluation
