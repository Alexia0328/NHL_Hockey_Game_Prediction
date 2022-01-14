import pickle
from re import X
import pandas as pd
import os, sys
path_name = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(path_name)
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from utils.utilities import generate_roc_curve, generate_goal_rate_curve, generate_cumu_goal_curve
from utils.utilities import get_shot_array, get_goal_rate_shot_percentile
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from matplotlib.gridspec import GridSpec
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


RANDOM_SEED = 1337



datafile_std = '../data/ms2Q4_STD.csv'


def data_load(dataset_name, columns=None):
    dataset=pd.read_csv(dataset_name)
    if columns is not None:
        dataset = dataset[columns]

    dataset.dropna(axis = 0, inplace = True)
    X = dataset.drop(['Is goal'], axis=1)
    y = dataset[['Is goal']]
    # print(X.head())
    # print(y.head())

    return train_test_split(X, y)

def generate_calibration_display(clf, arr_x_test, arr_y_test, arr_y_pred, arr_y_prob, plot_name, label_names, save = False, show = True): 
    fig, ax1 = plt.subplots()
    # gs = GridSpec(4, 2)
    # ax_calibration_curve = fig.add_subplot(111)
    for i in range(len(clf)):
        display = CalibrationDisplay.from_estimator(
        clf[i],
        arr_x_test[i],
        arr_y_test[i],
        n_bins=20,
        name=label_names[i],
        ax=ax1,
    )
    # ax1.grid()
    ax1.set_title("Reliability Curve")
    plt.legend(loc="upper left")

    # plt.title('Calibration curves')

    if save:
        try: 
            # plt.savefig("../../figs/" + plot_name)
            # plt.savefig("../../figs/" + 'calibration_curve.png')
            plt.savefig(plot_name)
        except: 
            print("failed to save plot {plot_name}")
    if show: 
        plt.show()

models = ['rbf_svc.sav', 'random_forest_300_6_10.sav', "random_forest_300_9.sav", "random_forest_500_6.sav"]
labels = [x[:-4] for x in models]

arr_y_test, arr_x_test, arr_y_pred, arr_prob, arr_shot_percentile, arr_goal_rate, arr_goal_cumu, arr_clf = [], [], [], [], [], [], [], []
filepath = '../models/saved/'

for i, model in enumerate(models): 

    file = filepath + model
    loaded_model = pickle.load(open(file, 'rb'))

    _, x_test, _, y_test = data_load(datafile_std)
    y_pred = loaded_model.predict(x_test)

    y_prob = loaded_model.predict_proba(x_test)
    arr_y_test.append(y_test)
    arr_x_test.append(x_test)
    arr_y_pred.append(y_pred)
    arr_prob.append(y_prob)
    arr_clf.append(loaded_model)
    shots_array = get_shot_array(y_prob, y_test)
    shot_percentile, goal_rate, goal_cumu = get_goal_rate_shot_percentile(shots_array, bin_size=1)

    arr_shot_percentile.append(shot_percentile)
    arr_goal_rate.append(goal_rate)
    arr_goal_cumu.append(goal_cumu)


##generate_roc_curve(arr_y_test, arr_y_pred, arr_prob, label_names = labels, plot_title = 'roc_curve', save = False)
#generate_goal_rate_curve(arr_shot_percentile, arr_goal_rate, label_names = labels, plot_title='Goal_Rate', save = False)
#generate_cumu_goal_curve(arr_shot_percentile, arr_goal_cumu, label_names = labels, plot_title='Cumulative_goals', save = False)
generate_calibration_display(arr_clf, arr_x_test, arr_y_test, y_pred, y_prob, plot_name='Reliability_curve', label_names = labels, save = False)   

    

