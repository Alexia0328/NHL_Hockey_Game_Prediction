import os.path
import sys 
path_name = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(path_name)
import train
from xgboost import XGBClassifier
from sklearn import metrics
from utils.utilities import generate_roc_curve, generate_goal_rate_curve, generate_cumu_goal_curve, generate_calibration_display
from utils.utilities import get_shot_array, get_goal_rate_shot_percentile

data_list =  [['Distance from net', 'Angle from net', 'Is goal']]
label_names = ['xgb_distance_angle']

arr_y_test, arr_x_test, arr_y_pred, arr_prob, arr_shot_percentile, arr_goal_rate, arr_goal_cumu = [], [], [], [], [], [], []

for i, data in enumerate(data_list): 
    print(f"-------------------------{data}---------------------------------")
    clf = XGBClassifier(objective= 'binary:logistic')
    y_pred, y_prob, y_test, x_test, clf = train.train(clf, columns = data,  model_name = label_names[i], use_comet=False, save_model=False)
    print('Confusion Matrix:')
    print(metrics.confusion_matrix(y_test, y_pred))
    arr_y_test.append(y_test)
    arr_x_test.append(x_test)
    arr_y_pred.append(y_pred)
    arr_prob.append(y_prob)

    shots_array = get_shot_array(y_prob, y_test)
    shot_percentile, goal_rate, goal_cumu = get_goal_rate_shot_percentile(shots_array, bin_size=1)

    arr_shot_percentile.append(shot_percentile)
    arr_goal_rate.append(goal_rate)
    arr_goal_cumu.append(goal_cumu)
    print("----------------------------------------------------------------")

generate_roc_curve(arr_y_test, arr_y_pred, arr_prob, label_names = label_names, plot_title = 'roc_curve_xgb_base', save = False)
# generate_goal_rate_curve(arr_shot_percentile, arr_goal_rate, label_names = label_names, plot_title='Goal_Rate_xgb_base', save = True)
# generate_cumu_goal_curve(arr_shot_percentile, arr_goal_cumu, label_names = label_names, plot_title='Cumulative_goals_xgb_base', save = True)
# generate_calibration_display(clf, x_test, y_test, y_pred, y_prob, plot_name='Reliability_curve_xgb_base', label_names = label_names, save = True)
