from comet_ml import Experiment
import os
import sys
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, roc_auc_score


import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings("ignore")


load_dotenv()
DATA_PATH = os.path.abspath(os.path.dirname(__file__)) + '/data/'
RANDOM_SEED = 1337


def load_data(columns=None, scaler='standard', split=0.2) -> tuple: 
    """[loads the data -> scales it according to the speecified scaler -> returns train-val split]

    Args:
        columns(array, optional): [name of columns to select from the data]. Defaults to all columns
        scaler (str, optional): [choice of scalar]. Defaults to 'standard'.
        split (float, optional): [training and validation split]. Defaults to 0.2

    Returns: 
        x_train, x_val, y_train, y_val, x_test
    """
    train = pd.read_csv(os.path.join(DATA_PATH, "ms2Q4_STD.csv"))
    # test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"), index_col = 0)
    train.dropna(axis = 0, inplace = True)

    features = list(train.columns)
    features.remove('Is goal')
    #print(f"fetures: {features}")
    train_values = train[features].values

    # if scaler == 'standard': 
    #     scaler = StandardScaler()
    #     train_values = scaler.fit_transform(train_values)
    # if scaler == 'minmax': 
    #     scaler = MinMaxScaler()
    #     train_values = scaler.fit_transform(train_values)
    # else: 
    #     pass 

    train_scaled = pd.DataFrame(columns = features, data = train_values)
    y = train[['Is goal']]

    return train_test_split(train_scaled, y, test_size = split, random_state = RANDOM_SEED)
        


def train(clf, model_name, columns = None, use_comet=False, save_model=False, verbose = True):

    if use_comet: 
        exp = Experiment(
            api_key=os.environ.get('COMET_API_KEY'),
            project_name ='milestone-2',
            workspace = 'maskedviper',
            auto_output_logging = "default"
        )

    X_train, X_val, y_train, y_val= load_data(columns)

    clf = clf
    params = clf.get_params()
    clf.fit(X_train, y_train)

    # print(type(clf).__name__)
    MODEL_PATH = os.path.abspath(os.path.dirname(__file__)) + '/models/saved/' + type(clf).__name__ + "/"
    print(MODEL_PATH)

    y_pred_train = clf.predict(X_train)
    


    

    if save_model: 
        # filename = "models/saved/" + type(clf).__name__ + "/" + model_name + ".pkl"
        file = model_name + ".pkl"
        filename = os.path.join(MODEL_PATH, file)
        print(filename)
        # pickle.dump(clf, open(filename, 'wb'))
        try: 
            pickle.dump(clf, open(filename, 'wb'))
        except IOError as e: 
            print(f"I/O error {e.errno}: {e.strerror}")
        except: 
            print('Unexpected error ', sys.exec_info()[0])

    y_pred = clf.predict(X_val)   
    f1 = f1_score(y_val, y_pred)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    roc_score = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    
    metrics = {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall, "roc_score": roc_score, "report": report}

    params = {
        'random_state': RANDOM_SEED, 
        'model_type': model_name,
        'scaler': 'standard',
        'param_grid': str(params), 
    }
    filename = model_name + ".sav"
    import pickle 
    pickle.dump(clf, open(filename, 'wb'))
    if use_comet: 
        exp.log_parameters(params)
        exp.log_metrics(metrics)
        exp.log_model(model_name, filename)
        # exp.add_tag(model_name)

    if verbose: 
        print("#------------------------Training Report-------------------------------#") 
        print("#                      Classification Report                           #")
        print(classification_report(y_train, y_pred_train)) 
        print("#----------------------------------------------------------------------#")
        print(f"ROC AUC score: f{roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])}")
        print("------------------------------------------------------------------------")
        print("#------------------------Validation Report-----------------------------#") 
        print("#                      Classification Report                           #")
        print(report)
        print("#----------------------------------------------------------------------#")
        print(f"roc_auc_score {roc_score}")
        print("---------------------------*************-------------------------------")
        #print(f"{model_name} | Accuracy: {accuracy} f1: {f1}  recall: {recall} precision: {precision}")

    y_prob = clf.predict_proba(X_val)
    return y_pred, y_prob, y_val, X_val, clf

