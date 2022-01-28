import numpy as np

from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve, CalibrationDisplay

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

def get_shot_array(y_prob:np.ndarray, y_test:np.array)-> np.ndarray: 
    """[generates 2-D ndarray to be used for get_goal_rate_shot_percentile()]

    Args:
        y_prob (np.ndarray): [2D predictions: [no_goal goal]] for each shot]
        y_test (np.array): [actual labels for goal/no_goal]

    Returns:
        np.np.ndarray: [2D array sorted by probabilities]
    """
    shot_array = np.column_stack((y_prob[:,1], y_test))
    shot_array_sorted = shot_array[np.argsort(shot_array[:, 0])]
    return shot_array_sorted


def get_goal_rate_shot_percentile(shot_array:np.ndarray, bin_size:int):
    """
    Args: 
        shot_array (np.ndarray): [[probabilities], [shot_result]] -> sorted by probs
        bin_size (int): (step-size) size of the bin

    Summary: 
        the function first calculates the goal-rate and cumulative goals scored between the 
        percentiles: pi and (pi + bin_size)

    Returns:
        shot_percentile: array of bins 
        goal_rate: number of goals / total shots * 100 corresponding to each bin
        goal_cumu: cumulative goals corresponding to each bin
    """
    low_value = 0.0      # bin-attributes
    high_value = 0.0     # bin-attributes
    shot_percentile = []
    goal_rate = []
    goal_cumu = []
    shot_prob_array = np.array(shot_array[:,0])
    total_goals = np.count_nonzero(shot_array[:, 1] == 1)
    
    for i in range(0, 101, bin_size):
        p = np.percentile(shot_prob_array, i)
        cumu_goal_count = 0 
        if i == 0:
            low_value = p
        else:
            shot_percentile.append(i-1)
            high_value = p
            goal_count = 0
            no_goal_count = 0
            
            for shot in shot_array:
                #calculating goal_rate
                if shot[0] >= low_value and shot[0] < high_value:
                    if shot[1] == 0:
                        no_goal_count += 1
                    elif shot[1] == 1:
                        goal_count += 1
                
                #calculating cumulative percentage of goals 
                if (shot[0] >= p and shot[1] == 1): 
                    cumu_goal_count += 1 
 
            try:
                goal_rate.append(goal_count / (goal_count + no_goal_count) * 100)
                goal_cumu.append((cumu_goal_count/ total_goals)*100)
            except ZeroDivisionError:
                goal_rate.append(0)
                goal_cumu.append(0)
            low_value = high_value

    return shot_percentile, goal_rate, goal_cumu


def generate_roc_curve(arr_y_test, arr_y_pred, arr_y_prob, plot_title, label_names, save = False, show = True) -> None: 
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1],'r--', label = 'Random Baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC_Curve for" +  plot_title)

    for i in range(len(arr_y_test)): 
        logit_roc_auc = roc_auc_score(arr_y_test[i], arr_y_prob[i][:, 1])
        fpr, tpr, thresholds = roc_curve(arr_y_test[i], arr_y_prob[i][:, 1])
        plt.plot(fpr, tpr, label = str(label_names[i]) + f"  score: {logit_roc_auc:.4f}", alpha = 0.7)


    
    plt.title('ROC-AUC Curve')
    plt.legend(loc="lower right")
    # plt.savefig("../../figs/" + plot_title)
    if save: 
        try: 
            # plt.savefig("../../figs/" + plot_title)
            plt.savefig("figs/" + plot_title)
        except: 
            print("failed to save plot {plot_title}")    
    if show: 
        plt.show()    

def generate_goal_rate_curve(arr_shot_percentile, arr_goal_rate, plot_title,label_names, save = False, show = True) -> None: 
    plt.figure(figsize = (7, 7))
    plt.plot([0, 100], [0, 100],'r--', label = 'Random Baseline')
    plt.xlim([100, 0])
    plt.ylim([0, 100])

    for i in range(len(arr_shot_percentile)): 
        plt.plot(arr_shot_percentile[i], arr_goal_rate[i], label = label_names[i])

    plt.xlabel('Shot Probability Model Percentile')
    plt.ylabel('Goals / Shots + Goals in %')
    plt.title('Goal_Rate Curve')
    plt.legend(loc="upper right")


    if save:
        try: 
            # plt.savefig("../../figs/" + plot_title)
            plt.savefig("figs/" + plot_title)
        except: 
            print("failed to save plot {plot_title}")
    if show: 
        plt.show()

def generate_cumu_goal_curve(arr_shot_percentile, arr_goals_cumu, plot_title, label_names, save = False, show = True): 
    plt.figure(figsize = (7, 7))
    plt.plot([0, 100], [100, 0],'r--', label = 'Random Baseline')
    plt.xlim([100, 00])
    plt.ylim([0, 100])

    for i in range(len(arr_shot_percentile)): 
        plt.plot(arr_shot_percentile[i], arr_goals_cumu[i], label = label_names[i])

    plt.xlabel('Shot Probability Model Percentile')
    plt.ylabel('Proportion in %')
    plt.title('Cumulative % of goals Curve')
    plt.legend(loc="lower right")

    if save:
        try: 
            # plt.savefig("../../figs/" + plot_title)
            plt.savefig("figs/" + plot_title)
        except: 
            print("failed to save plot {plot_title}")
    if show: 
        plt.show()

def generate_calibration_display(clf, arr_x_test, arr_y_test, arr_y_pred, arr_y_prob, plot_name, label_names, save = False, show = True): 

    disp = CalibrationDisplay.from_estimator(clf, arr_x_test, arr_y_test, n_bins=20)

    plt.title('Calibration curves')

    if save:
        try: 
            # plt.savefig("../../figs/" + plot_name)
            # plt.savefig("../../figs/" + 'calibration_curve.png')
            plt.savefig("figs/" + plot_name)
        except: 
            print("failed to save plot {plot_name}")
    if show: 
        plt.show()
