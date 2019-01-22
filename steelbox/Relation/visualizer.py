import matplotlib.pyplot as plt
import pickle
import numpy as np

def getdata(path='dataset.p'):
    with open(path,'rb') as f:
        images = pickle.load(f)
        questions = pickle.load(f)
        answers = pickle.load(f)
    #plt.imshow(images[0])
    #plt.show()
    images = np.array(images).transpose([0,3,1,2])
    print(np.array(images).shape)
    return images,questions,answers

def getimage(path='dataset.p'):
    with open(path,'rb') as f:
        images = pickle.load(f)
        #questions = pickle.load(f)
        #answers = pickle.load(f)
    #plt.imshow(images[0])
    #plt.show()
    images = images.detach().numpy()
    print(images.shape)
    images = images.transpose([0,2,3,1])
    print(np.array(images).shape)
    return images

if __name__ == '__main__':
    images = getimage('relation_dec.p')
    print(images.shape)
    plt.imshow(images[0]*255)
    plt.show()

