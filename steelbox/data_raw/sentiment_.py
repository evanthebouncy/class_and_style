import numpy as np

def gen_data(data_dir):
    the_path = data_dir + "/sentiment/"
    print ("loading sentiment from ", the_path)
    train_images = np.load(the_path+'train/embeddings.npy')
    train_labels = np.load(the_path + 'train/labels.npy')
    test_images = np.load(the_path + 'test/embeddings.npy')
    test_labels = np.load(the_path + 'test/labels.npy')
    return train_images, train_labels,test_images, test_labels

if __name__ == '__main__':
    train_images,train_labels,test_images,test_labels = gen_data('.')
    print(train_images.shape,train_labels.shape,test_images.shape,test_labels.shape)
    print(train_images[0])