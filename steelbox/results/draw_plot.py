import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np

def getresult():
    all_results = []
    for line in open('results/eval_subset_results_20190121T1207.jsonl'):
        try:
            all_results.append(json.loads(line))
        except:
            pass
    return all_results

def draw_plot(model_name,all_results):
    tt=0
    tot=0
    df = pd.DataFrame(columns = ['random','tiers','kmeans','tiers_anneal','kmeans_anneal','random_anneal',
                                 'random_size','tiers_size','kmeans_size','tiers_anneal_size','kmeans_anneal_size','random_anneal_size'])
    for result in all_results:
        if result['model_name'] != model_name:
            continue
        if result['model_name'] not in ['LGR', 'FC','CNN','DTREE','SVMrbf','SVMLin','EKNN','RFOREST']:
            print('wtf?????')
        if(result['num_samples'] >30000):
            tt=tt+1
        tot = tot+1
        df=df.append({result['subset_name']:(result['score_m_m'],result['num_samples'])},ignore_index=True)
        #df=df.append({result['subset_name']+'_size':result['num_samples']},ignore_index=True)
        #print(df)
    print(df['random'].dropna().values)

    #for subset_name in ['random','tiers','kmeans','tiers_anneal','kmeans_anneal','random_anneal']:
    for subset_name in ['random','kmeans']:
        value = df[subset_name].dropna().values
        sorted_value = sorted(value, key=lambda tup: tup[1])
        list_of_lists = np.array([list(elem) for elem in sorted_value])
        if(len(list_of_lists)==0):
            continue
        plt.plot(np.log(list_of_lists[:,1]),list_of_lists[:,0],label=subset_name)
    plt.legend()
    plt.show()
    print(tt,tot)

if __name__ == '__main__':
    print(len(getresult()))
    draw_plot('LGR',getresult())