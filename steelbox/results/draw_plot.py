import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import pylab

def getresult():
    all_results = []
    for line in open('results/eval_subset2_mnist_results_nnonly20190122T1700.jsonl'):
        try:
            all_results.append(json.loads(line))
        except:
            pass
    return all_results

def smoth(a):
    b=np.array([[]])
    for i in range(len(a)):
        if i == 0:
            b = np.array([a[i]])
        elif a[i,1] == a[i-1,1]:
            b[len(b)-1,0] = (b[len(b)-1,0]+a[i,0])/2.0
        else:
            b=np.append(b,np.array([a[i]]),axis=0)
    return b
def draw_plot(model_name,all_results):
    tt=0
    tot=0
    df = pd.DataFrame(columns = ['random','tiers','kmeans','tiers_anneal','kmeans_anneal','random_anneal',
                                 'random_size','tiers_size','kmeans_size','tiers_anneal_size','kmeans_anneal_size','random_anneal_size'])
    for result in all_results:
        if 'score_m_m' not in result.keys():
            continue

        if result['model'] != model_name:
            continue
        if result['model'] not in ['LGR', 'FC','CNN','DTREE','SVMrbf','SVMLin','EKNN','RFOREST']:
            print('wtf?????')
        print('score:', result['score_m_m'], result['num_samples'],result['subset_name'])
        if(result['num_samples'] >30000):
            tt=tt+1
        tot = tot+1
        df=df.append({result['subset_name']:(result['score_m_m'],result['num_samples'])},ignore_index=True)
        #df=df.append({result['subset_name']+'_size':result['num_samples']},ignore_index=True)
        #print(df)
    #print(df['random'].dropna().values)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    max_acc = 0

    for subset_name in ['random','tiers','kmeans','tiers_anneal','kmeans_anneal','random_anneal']:
    #for subset_name in ['random','kmeans']:
        #value = df[subset_name].dropna().values
        #print(df[subset_name])

        value = df[subset_name].dropna().values

        #print(value)
        #continue
        sorted_value = sorted(value, key=lambda tup: tup[1])
        list_of_lists = np.array([list(elem) for elem in sorted_value])
        if(len(list_of_lists)==0):
            continue
        print(list_of_lists)
        list_of_lists = smoth(list_of_lists)
        print(list_of_lists)

        line = ax.plot(list_of_lists[:,1],list_of_lists[:,0],label=subset_name)
        #line = ax.scatter(list_of_lists[:, 1], list_of_lists[:, 0], label=subset_name)
        max_acc = list_of_lists[len(list_of_lists)-1,0]

        print(tt,tot)
    ax.plot(np.arange(100,60000),max_acc*np.ones(60000-100,dtype=float))
    ax.set_xscale('log')
    plt.legend()
    #pylab.show()
    pylab.savefig('MNISTplot/mnist_'+model_name)


if __name__ == '__main__':
    print(len(getresult()))
    for models in ['FC','CNN', 'LGR']:
        draw_plot(models,getresult())