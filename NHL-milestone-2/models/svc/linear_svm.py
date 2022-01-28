import os.path
import sys

path_name = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(path_name)
import train
from utils.utilities import generate_roc_curve
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pickle


clf = LinearSVC(C = 1.0, penalty = 'l1', dual = False, tol = 1e-3,  max_iter= 300, random_state = 1337)
y_pred, y_prob, y_val, x_val, clf = train.train(clf, model_name = 'linear_svc', use_comet= True)

pickle.dump(clf, open('linear_svm_c1.sav', 'wb'))
# y_pred_train = clf.predict(x_train)
# y_pred_test = clf.predict(x_test)
# report_train = classification_report(y_train, y_pred_train)
# report_test = classification_report(y_test, y_pred_test)
# print("#------------------------Training Report-------------------------------#") 
# print("#                      Classification Report                           #")
# print(report_train) 
# print()
# print("#------------------------Validation Report-----------------------------#") 
# print("#                      Classification Report                           #")
# print(report_test)
# print("#----------------------------------------------------------------------#")
# print("---------------------------*************-------------------------------")