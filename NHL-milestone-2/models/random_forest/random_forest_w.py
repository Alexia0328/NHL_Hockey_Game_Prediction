#------------------------------0- weighted random forest -----------------------------------# 
## Here we attempt to utilize the parameter: class_weight to counter the label imbalance   ## 
## After finding the setting that has improved metric (recall for label 1), we try the clf ##
#-------------------------------------------------------------------------------------------#
import numpy as np
import os.path
import sys
from comet_ml.loggers.xgboost_logger import _xgboost_train
from scipy.sparse.construct import random 
path_name = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(path_name)
from sklearn.ensemble import RandomForestClassifier
import train
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

import matplotlib.pyplot as plt
import pickle

max_depth = [2, 3, 4, 5, 6]
precision_1_train, precision_1_test, precision_0_train, precision_0_test = [], [], [], []
recall_1_train, recall_1_test, recall_0_train, recall_0_test = [], [], [], []
f1_train = []
f1_test = []



clf = RandomForestClassifier(n_estimators= 300, max_depth = 6, max_features = 10, class_weight= 'balanced', random_state= 1337) 
y_pred, y_prob, y_val, x_val, clf = train.train(clf, model_name= "random_forest_300_6_10", use_comet= True)

#pickle.dump(clf, open('random_forest_300_9.sav','wb') )
# prob_0 = y_prob[:, 0]
# prob_1 = y_prob[:, 1]

# plt.hist(prob_0, bins = 'auto', label = 'label 0')
# plt.hist(prob_1, bins = 'auto', label = 'label 1')

# for depth in max_depth: 
#     clf = RandomForestClassifier(n_estimators= 300, max_depth= depth, class_weight= 'balanced', random_state= 1337)
    
#     clf.fit(x_train, y_train)

#     y_pred_train = clf.predict(x_train)
#     prec_train = precision_score(y_train, y_pred_train, average=None)
#     precision_0_train.append(prec_train[0])
#     precision_1_train.append(prec_train[1])
#     recall_train = recall_score(y_train, y_pred_train, average= None)
#     recall_0_train.append(recall_train[0])
#     recall_1_train.append(recall_train[1])
#     f1_train.append(f1_score(y_train, y_pred_train))

#     y_pred_test = clf.predict(x_test)
#     prec_test = precision_score(y_test, y_pred_test, average= None)
#     precision_0_test.append(prec_test[0])
#     precision_1_test.append(prec_test[1])
#     recall_test = recall_score(y_test, y_pred_test,average= None)
#     recall_0_test.append(recall_test[0])
#     recall_1_test.append(recall_test[1])
#     f1_test.append(f1_score(y_test, y_pred_test))


# fig, axs = plt.subplots(1, 3, figsize = (20, 9))

# axs[0].scatter(precision_1_train, recall_1_train, color = 'green', s = np.array(max_depth)**3,  label = 'label 1 training')
# axs[0].scatter(precision_1_test, recall_1_test, color = 'red',s = np.array(max_depth)**3, label = 'label 1 test')
# axs[0].set_xlabel('Precision')
# axs[0].set_ylabel('Recall')
# axs[0].set_title("Label 1 (marker size is proportional to depth)")
# axs[0].legend(loc = 'upper left')

# axs[1].scatter(precision_0_train, recall_0_train, color = 'green', s = np.array(max_depth)**3, label = 'label 0 training')
# axs[1].scatter(precision_0_test, recall_1_test, color = 'red', s = np.array(max_depth)**3, label = 'label 0 test')
# axs[1].set_xlabel('Precision')
# axs[1].set_ylabel('Recall')
# axs[1].set_title("Label 0 (marker size is proportional to depth)")
# axs[1].legend(loc = 'upper left')

# axs[2].plot(max_depth, f1_train, color = 'green', label = 'f1 score: training')
# axs[2].plot(max_depth, f1_test, color = 'red', label = 'f1 score: testing')
# axs[2].set_xlabel('Max Depth')
# axs[2].set_ylabel('f1 score')
# axs[2].set_title("f1 score wrt diff depths")
# axs[2].legend(loc = 'upper left')

# plt.title("Effect of change in depth")
# plt.show()