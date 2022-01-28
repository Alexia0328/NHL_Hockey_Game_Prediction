import os.path
import sys 
path_name = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(path_name)
from sklearn.ensemble import RandomForestClassifier
import train
from utils.utilities import generate_roc_curve


n_estimators = [100, 200, 300, 400, 500]
arr_y_test, arr_x_test, arr_y_pred, arr_prob, arr_shot_percentile, arr_goal_rate, arr_goal_cumu = [], [], [], [], [], [], []

for n in n_estimators: 
    print(f"------------------------n_estimators = {n}-------------------------------")
    clf = RandomForestClassifier(n_estimators=n, max_depth = 2, random_state=1337)

    y_pred, y_prob, y_test, x_test = train.train(clf, model_name = 'random_forest_n_estimators' + str(n)) 
    arr_y_test.append(y_test)
    arr_x_test.append(x_test)
    arr_y_pred.append(y_pred)
    arr_prob.append(y_prob)


generate_roc_curve(arr_y_test, arr_y_pred, arr_prob, label_names = [f"depth = {x}" for x in n_estimators], plot_title = 'ROC curve with varying estimators')


