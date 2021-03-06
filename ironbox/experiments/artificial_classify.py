# ======================= HELPERS =====================
# compute the accuracy of using a classifier trained on the subset
def get_acc(clf, data_holder, X_t, Y_t):
  clf.learn(data_holder)
  return clf.evaluate((X_t,Y_t)) 

# ======================== RUNNING THE EXPERIMENT ===========================
def run_subset_size(clf, X_tr, Y_tr, X_t, Y_t, n_sub, ae):
  '''
    clf : the classifier, say a fully connected neural network
    X_tr, Y_tr, X_t, Y_t : self explainatory
    n_sub : number of subset
    ae : the auto-encoder
  '''
  ret_dict = {
        'meta' : {
          'name' : 'artificial_classify',
          'n_sub' : n_sub,
          'evaluation_model' : clf().name,
        },
        'results' : {},
      }

  result_dict = ret_dict['results']

  print ("================ running for ", n_sub, clf().name, "====================")
  data_holder = DataHolder(X_tr, Y_tr, [1.0] * len(X_tr))
  all_error = get_acc(clf(), data_holder, X_t, Y_t)
  print ( "all error : ", all_error)
  result_dict['all'] = all_error

  r_idx = sub_select_random(X_tr, Y_tr, n_sub)
  data_holder = DataHolder(X_tr[r_idx], Y_tr[r_idx], [1.0] * n_sub)
  rand_error = get_acc(clf(), data_holder, X_t, Y_t)
  print ( "rand error : ", rand_error)
  result_dict['rand'] = rand_error

  sub_idxs = sub_select_cluster(X_tr, n_sub, ae) # this cluster does not depend on Y
  data_holder = DataHolder(X_tr[sub_idxs], Y_tr[sub_idxs], [1.0] * n_sub)
  clus = get_acc(clf(), data_holder, X_t,Y_t)
  print ( "cluster error : ", clus )
  result_dict['cluster'] = clus

  sub_idxs = sub_select_label_cluster(X_tr, Y_tr, n_sub, ae) # this cluster does depend on Y
  data_holder = DataHolder(X_tr[sub_idxs], Y_tr[sub_idxs], [1.0] * n_sub)
  clus = get_acc(clf(), data_holder, X_t,Y_t)
  print ( "label cluster error : ", clus )
  result_dict['label cluster'] = clus

  sub_idxs = sub_select_knn(X_tr, Y_tr, n_sub, ae)
  data_holder = DataHolder(X_tr[sub_idxs], Y_tr[sub_idxs], [1.0] * n_sub)
  knnn = get_acc(clf(), data_holder, X_t,Y_t)
  print ( "knn error : ", knnn )
  result_dict['knn'] = knnn

  sub_idxs = sub_select_knn_dueling(X_tr, Y_tr, n_sub, ae)
  data_holder = DataHolder(X_tr[sub_idxs], Y_tr[sub_idxs], [1.0] * n_sub)
  knnn_dueling = get_acc(clf(), data_holder, X_t,Y_t)
  print ( "knn dueling error : ", knnn_dueling )
  result_dict['knn_dueling'] = knnn_dueling

  return ret_dict

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

  from subset_selection.subset_selection import sub_select_cluster,\
                                                sub_select_knn,\
                                                sub_select_label_cluster,\
                                                sub_select_random,\
                                                sub_select_knn_dueling

  from evaluation_models.classify_fcnet import FCNet

  CLFS = {
      'fcnet' : lambda : FCNet(20, 2).cuda(),
  }

  # ================= parse some shit ! =================
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("eval_model_name")
  parser.add_argument("subset_size", type=int)
  args = parser.parse_args()

  # =============== run some shit on the parsed shit =================
  eval_model_name = args.eval_model_name
  assert eval_model_name in CLFS
  subset_size = args.subset_size


  result_dict = run_subset_size(CLFS[eval_model_name], 
                                X_tr, Y_tr, X_t, Y_t, subset_size, ae)
  print (result_dict)



