from comet_ml import API
import pickle
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from utils.utilities import generate_roc_curve, generate_goal_rate_curve, generate_cumu_goal_curve
from utils.utilities import get_shot_array, get_goal_rate_shot_percentile
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn import metrics
import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


api = API()
RANDOM_SEED = 1337

regular = True

if regular:
    datafile = 'data\ms2Q4_test_one_hot.csv'
    datafile_std = 'data\ms2Q4_test_STD.csv'
else:
    datafile = 'data\ms2Q4_test_one_hot_playoff.csv'
    datafile_std = 'data\ms2Q4_test_STD_playoff.csv'

def data_load(dataset_name, columns=None):
    dataset=pd.read_csv(dataset_name)
    if columns is not None:
        dataset = dataset[columns]
    print(dataset.shape)
    dataset.dropna(axis = 0, inplace = True)
    print(dataset.shape)
    X = dataset.drop(['Is goal'], axis=1)
    y = dataset[['Is goal']]
    # print(X.head())
    # print(y.head())
    return X, y

def getmetrices(y_test, y_pred):
    score = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    metric = {"accuracy": score, "f1": f1, "recall": recall, "precision": precision}
    print('metrices: ', metric)
    print(report)
    print('Confusion Matrix:')
    print(metrics.confusion_matrix(y_test, y_pred))

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
            plt.savefig("figs/" + plot_name)
        except: 
            print("failed to save plot {plot_name}")
    if show: 
        plt.show()

# Download a Registry Model:
model_dict = {
    'log-reg-dist': 'log_reg_dist.pkl',
    'log-reg-angle': 'log_reg_angle.pkl',
    'log-reg-dist-angle': 'log_reg_dist_angle.pkl',
    'xgb-optimum': 'xgb_lg_f1_25.pkl',
    'random-forest-300-9': 'random_forest_300_9.sav'
    }

data_list =  [['Distance from net', 'Is goal'],['Angle from net', 'Is goal'],['Distance from net', 'Angle from net', 'Is goal'],None, None]
label_names = ['lg_distance', 'lg_angle', 'lg_distance_angle', 'xgb_optimum', 'weighted_random_forest']
arr_y_test, arr_x_test, arr_y_pred, arr_prob, arr_shot_percentile, arr_goal_rate, arr_goal_cumu, arr_clf = [], [], [], [], [], [], [], []
filepath = 'models/test_download'
for idx, (model_name, file_name) in enumerate(model_dict.items()):
    print(f"-------------------------{model_name}---------------------------------")
    print(idx)
    print(model_name)
    print(file_name)
    # print(data_list[idx])
    api.download_registry_model("maskedviper", model_name, "1.0.0", output_path=filepath, expand=True)
    file = filepath + '/' + file_name
    loaded_model = pickle.load(open(file, 'rb'))
    if idx < 4:
        X, y = data_load(datafile, columns = data_list[idx])
    else:
        X, y = data_load(datafile_std, columns = data_list[idx])
    y_pred = loaded_model.predict(X)
    getmetrices(y, y_pred)
    y_prob = loaded_model.predict_proba(X)
    arr_y_test.append(y)
    arr_x_test.append(X)
    arr_y_pred.append(y_pred)
    arr_prob.append(y_prob)
    arr_clf.append(loaded_model)
    shots_array = get_shot_array(y_prob, y)
    shot_percentile, goal_rate, goal_cumu = get_goal_rate_shot_percentile(shots_array, bin_size=1)

    arr_shot_percentile.append(shot_percentile)
    arr_goal_rate.append(goal_rate)
    arr_goal_cumu.append(goal_cumu)

if regular:
    generate_roc_curve(arr_y_test, arr_y_pred, arr_prob, label_names = label_names, plot_title = 'roc_curve_reg_test', save = True)
    generate_goal_rate_curve(arr_shot_percentile, arr_goal_rate, label_names = label_names, plot_title='Goal_Rate_reg_test', save = True)
    generate_cumu_goal_curve(arr_shot_percentile, arr_goal_cumu, label_names = label_names, plot_title='Cumulative_goals_reg_test', save = True)
    generate_calibration_display(arr_clf, arr_x_test, arr_y_test, y_pred, y_prob, plot_name='Reliability_curve_reg_test', label_names = label_names, save = True)   
else:
    generate_roc_curve(arr_y_test, arr_y_pred, arr_prob, label_names = label_names, plot_title = 'roc_curve_play_test', save = True)
    generate_goal_rate_curve(arr_shot_percentile, arr_goal_rate, label_names = label_names, plot_title='Goal_Rate_play_test', save = True)
    generate_cumu_goal_curve(arr_shot_percentile, arr_goal_cumu, label_names = label_names, plot_title='Cumulative_goals_play_test', save = True)
    generate_calibration_display(arr_clf, arr_x_test, arr_y_test, y_pred, y_prob, plot_name='Reliability_curve_play_test', label_names = label_names, save = True)
    