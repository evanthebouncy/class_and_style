import argparse

def get_data(filename):
    data = __import__(filename)
    get_data_ = getattr(data,'make_dataset')
    return get_data_(200)

def select_data(algorithm_name,X,Y,leng=10,ratio=0.5,chosepoint=10):
    alg = __import__(algorithm_name)
    func = getattr(alg,'select')
    return func(leng,ratio,chosepoint,X,Y,1)

if __name__ == '__main__':
    filename = 'artificial'
    algorithm_name = 'rec_annealing'
    X,Y = get_data(filename)
    print(select_data(algorithm_name,X,Y))

