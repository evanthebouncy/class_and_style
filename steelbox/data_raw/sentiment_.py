import numpy as np

def gen_data(data_dir):
    the_path = data_dir + "/sentiment/"
    print ("loading sentiment from ", the_path)
    train = np.load(the_path + 'sst_binary_bert_large_avgpool_layer-2.npz')
    test  = np.load(the_path + 'sst_binary_test_bert_large_avgpool_layer-2.npz')
    return train['embeddings'], train['labels'], test['embeddings'], test['labels']

if __name__ == '__main__':
    train_embs,train_labels,test_embs,test_labels = gen_data('.')
    print(train_embs.shape,train_labels.shape,test_embs.shape,test_labels.shape)
    print(train_embs[0])
    print(np.unique(train_labels))
