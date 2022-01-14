from comet_ml import Experiment
import os
import sys
import xgboost as xgb
import pandas as pd
import numpy as np
from numpy import loadtxt
from numpy import sort
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
import optuna
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_optimization_history
import shap
# use feature importance for feature selection
from sklearn.feature_selection import SelectFromModel
from utils.utilities import generate_roc_curve, generate_goal_rate_curve, generate_cumu_goal_curve, generate_calibration_display
from utils.utilities import get_shot_array, get_goal_rate_shot_percentile

# Create an experiment with your api key
# exp = Experiment(
#     api_key=os.environ.get('COMET_API_KEY'),
#     project_name ='milestone-2',
#     workspace = 'maskedviper',
#     auto_output_logging = "default"
# )

model_name = 'xgb_optimum'
dataset_2=pd.read_csv('data\ms2Q4_one_hot.csv')
X_2 = dataset_2.iloc[:, :-1]
y_2 = dataset_2.iloc[: , -1]
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2,y_2,test_size=0.2, random_state=0)

RANDOM_SEED = 42
# 10-fold CV
kfolds = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

# Helper function so that it can be reused
def tune(objective):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)
    fig1 = plot_optimization_history(study, target_name='f1 score')
    fig2 = plot_param_importances(study, target_name='f1 score')
    # fig1.show()
    fig1.write_image('figs\opt_hist_f1_25.png')
    # fig2.show()
    fig2.write_image('figs\param_imp_f1_25.png')
    params = study.best_params
    best_score = study.best_value
    print(f"Best score: {best_score}\n")
    print(f"Optimized parameters: {params}\n")
    return params

# XGBoost Objective for hyperparameter tuning

def xgboost_objective(trial):
    _n_estimators = trial.suggest_int("n_estimators", 10, 200)
    _max_depth = trial.suggest_int("max_depth", 4, 10)
    _alpha = trial.suggest_int("alpha", 3, 15)
    _colsample_bytree = trial.suggest_float("colsample_bytree", 0.3, 1.0)
    _learning_rate = trial.suggest_categorical("learning_rate", [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3])


    model = XGBClassifier(objective ='binary:logistic', colsample_bytree = _colsample_bytree, learning_rate = _learning_rate,
                    max_depth = _max_depth, alpha = _alpha, n_estimators = _n_estimators, verbosity=0, use_label_encoder=False)
    

    scores = cross_val_score(
        model, X_train_2, y_train_2, cv=kfolds, scoring="f1"
    )
    return scores.mean()

def feature_selection(loaded_model, xgboost_params):
    thresholds = sort(loaded_model.feature_importances_)
    feature_count = []
    feature_accuracy = []
    # print(thresholds)
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(loaded_model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train_2)
        feature_idx = selection.get_support()
        feature_name = X_train_2.columns[feature_idx]
        # train model
        selection_model = XGBClassifier(objective ='binary:logistic', **xgboost_params, use_label_encoder=False)
        selection_model.fit(select_X_train, y_train_2)
        # eval model
        select_X_test = selection.transform(X_test_2)
        y_pred = selection_model.predict(select_X_test)
        accuracy = accuracy_score(y_test_2, y_pred)
        feature_count.append(select_X_train.shape[1])
        feature_accuracy.append(accuracy)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
        print("Features: ", feature_name)
        print(classification_report(y_test_2, y_pred))

    plt.figure()
    plt.plot(feature_count, feature_accuracy, '-o')
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for number of features during feature selection')
    plt.savefig(r'figs\sfm_feature_sel')


# Get the parameters for best model after Hyper parameter tuning
# xgboost_params = tune(xgboost_objective)

# best_model = XGBClassifier(objective ='binary:logistic', **xgboost_params, use_label_encoder=False)
# best_model.fit(X_train_2, y_train_2)
# pickle.dump(best_model, open(r'models\xg\xgb_lg_f1_25.pkl', 'wb'))

# load model from file saved in above code
loaded_model = pickle.load(open(r'models\xg\xgb_lg_f1_25.pkl', 'rb'))
y_pred = loaded_model.predict(X_train_2)
y_prob = loaded_model.predict_proba(X_test_2)

# score = accuracy_score(y_test_2, y_pred)
# f1 = f1_score(y_test_2, y_pred)
# precision = precision_score(y_test_2, y_pred)
# recall = recall_score(y_test_2, y_pred)
# report = classification_report(y_test_2, y_pred)
report = classification_report(y_train_2, y_pred)

# print('Accuracy', score)
print(report)

"""
    For logging into commet after the saving the model

    Below parameters was obtained after the successfull execution of above code for xgboost hypeparameter tuning.
    Assigning them manually to avoid the long execution of hyperparameter tuning again while logging into comet.
    Please comment out 'xgboost_params' below if running the whole hyperparameter tuning again

    Returns: 
        x_train, x_val, y_train, y_val, x_test
"""
# Optimized parameters: {'n_estimators': 166, 'max_depth': 9, 'alpha': 13, 'colsample_bytree': 0.9943925059480794, 'learning_rate': 0.3}
xgboost_params = {'n_estimators': 166, 'max_depth': 9, 'alpha': 13, 'colsample_bytree': 0.9943925059480794, 'learning_rate': 0.3}
params = {
    'random_state': RANDOM_SEED, 
    'model_type': model_name,
    'param_grid': str(xgboost_params), 
}
    

# feature_selection(loaded_model, xgboost_params)


# SHAP to vizualize the feature importance 
# explainer = shap.Explainer(loaded_model)
# shap_values = explainer(X_train_2)
# fig = shap.plots.force(shap_values[0], matplotlib=True, show=False)
# plt.savefig('figs\pred_exp_force_f1_25.png')

label_names = []
label_names.append(model_name)

arr_y_test, arr_x_test, arr_y_pred, arr_prob, arr_shot_percentile, arr_goal_rate, arr_goal_cumu = [], [], [], [], [], [], []

arr_y_test.append(y_test_2)
arr_x_test.append(X_test_2)
arr_y_pred.append(y_pred)
arr_prob.append(y_prob)

shots_array = get_shot_array(y_prob, y_test_2)
shot_percentile, goal_rate, goal_cumu = get_goal_rate_shot_percentile(shots_array, bin_size=1)

arr_shot_percentile.append(shot_percentile)
arr_goal_rate.append(goal_rate)
arr_goal_cumu.append(goal_cumu)

# generate_roc_curve(arr_y_test, arr_y_pred, arr_prob, label_names = label_names, plot_title = 'roc_curve_xgb_tune', save = True)
# generate_goal_rate_curve(arr_shot_percentile, arr_goal_rate, label_names = label_names, plot_title='Goal_Rate_xgb_tune', save = True)
generate_cumu_goal_curve(arr_shot_percentile, arr_goal_cumu, label_names = label_names, plot_title='Cumulative_goals_xgb_tune', save = True)
# generate_calibration_display(loaded_model, X_test_2, y_test_2, y_pred, y_prob, plot_name='Reliability_curve_xgb_tune', label_names = label_names, save = True)


metrics = {"accuracy": score, "f1": f1, "recall": recall, "precision": precision, "report": report}

params = {
    'random_state': RANDOM_SEED, 
    'model_type': model_name,
    'scaler': 'standard',
    'param_grid': str(xgboost_params), 
}

# exp.log_parameters(params)
# exp.log_metrics(metrics)
# exp.log_model(model_name, r'models\xg\xgb_lg_f1_25.pkl')   