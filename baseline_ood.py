import pickle
import warnings
from inspect import isfunction
from multiprocessing import Process, Queue
from sklearn.preprocessing import StandardScaler

import numpy as np
# from algo import *
from sklearn.utils import shuffle as sk_shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, confusion_matrix, recall_score, make_scorer, accuracy_score, average_precision_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import shap
import matplotlib.pyplot as plt
# from utils3 import *
import pickle
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.cluster import KMeans
from sklearn.utils import shuffle as sk_shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, auc, confusion_matrix, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
import os 
import geopandas as gpd
import folium
from shapely.geometry import Point
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import shap
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score
import shap
from sklearn.utils import resample
import rasterio
# from metric import Feature_Filter
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import sys

warnings.filterwarnings("ignore")

class logiAlgo(LogisticRegression):
    DEFAULT_CONTINUOUS_BOOK = {'C':[0.5,1.0]}
    DEFAULT_DISCRETE_BOOK = {}
    DEFAULT_ENUM_BOOK = {
        'penalty':['l1','l2'], 
        'solver': ["liblinear", "saga"]
        }
    DEFAULT_STATIC_BOOK = {'max_iter':2000}
    
    def __init__(self, params):
        super().__init__(**params)
        self.params = params
        
    def predictor(self, X):   
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:,1]
    
class svmAlgo(SVC):
    DEFAULT_CONTINUOUS_BOOK = {'C':[0.5, 0.8]}
    DEFAULT_DISCRETE_BOOK = {} 
    DEFAULT_ENUM_BOOK = {'tol': [1e-3, 1e-4, 1e-5], 'kernel':['poly', 'rbf']} 
    DEFAULT_STATIC_BOOK = {'max_iter':2000, 'probability':True}

    def __init__(self, params):
        super().__init__(**params)
        self.params = params

    def predictor(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:,1]

class rfcAlgo(RandomForestClassifier):
    DEFAULT_CONTINUOUS_BOOK = {}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [10, 150], 'max_depth': [10, 50]}
    DEFAULT_ENUM_BOOK = {'criterion': ['gini', 'entropy']}
    DEFAULT_STATIC_BOOK = {} 
    
    def __init__(self, params):
        super().__init__(**params)
        self.params = params

    def predictor(self, X):

        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:,1]
    
class extAlgo(ExtraTreesClassifier):
    DEFAULT_CONTINUOUS_BOOK = {}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [10, 200], 'max_depth': [10, 50]}
    DEFAULT_ENUM_BOOK = {'criterion': ['gini', 'entropy']}
    DEFAULT_STATIC_BOOK = {}

    def __init__(self, params):
        super().__init__(**params)
        self.params = params

    def predictor(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:, 1]

def criterion_loss(mode, algo, X_val_fold, y_val_fold, y_train_fold = None):
    
    # the loss applied in training process
    class_weights = compute_class_weight('balanced', classes=np.unique(y_val_fold), y=y_val_fold)
    class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_val_fold), class_weights)}
    
    def weighted_cross_entropy(y_true, y_pred, weight = 'balanced', epsilon = 1e-7):
        sample_weights = compute_sample_weight(class_weight_dict, y_true)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(sample_weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
        return loss
                
    weighted_ce_scorer = make_scorer(weighted_cross_entropy, greater_is_better=False)
    return weighted_ce_scorer(algo, X_val_fold, y_val_fold)

def get_qhat(algo, X_val_fold, y_val_fold, alpha1, alpha2):
    probabilities = algo.predict_proba(X_val_fold)
    softmax_outputs_pos = probabilities[:,1]
    softmax_outputs_neg = probabilities[:,0]
    
    N1 = softmax_outputs_pos.shape[0]
    scores1=np.zeros(N1)

    for i in range(N1):
        if y_val_fold[i] == 1:
            scores1[i]= 1-softmax_outputs_pos[i]
        #else:
        #    scores1[i] = softmax_outputs_pos[i]
        
    non_zero_count1 = np.count_nonzero(scores1)
            
    val1 = np.quantile(scores1,np.ceil((N1+1)*(1-alpha1))/N1)


    N2 = softmax_outputs_neg.shape[0]
    scores2=np.zeros(N2)

    for i in range(N2):
        if y_val_fold[i] == 0:
            scores2[i]= 1-softmax_outputs_neg[i]
        #else:
        #    scores2[i] = softmax_outputs_neg[i]
        
    non_zero_count2 = np.count_nonzero(scores2)
        
    val2 = np.quantile(scores2,np.ceil((N2+1)*(1-alpha2))/N2)
    
    val1 = np.quantile(scores1,np.ceil((non_zero_count1+1)*(1-alpha1))/non_zero_count1)
    val2 = np.quantile(scores2,np.ceil((non_zero_count2+1)*(1-alpha2))/non_zero_count2)     
    
    return val1, val2

def get_valid_quantile(scores, counts, starting_alpha, increment=0.01, max_alpha=0.20, max_retries=15):
    retry_count = 0
    alpha = starting_alpha
    while retry_count < max_retries:
        try:
            quantile_value = np.quantile(scores, np.ceil((counts + 1) * (1 - alpha)) / counts)
            return quantile_value
        except ValueError as e:
            print(f"Quantiles must be in the range [0, 1]. Adjusting alpha. Retry {retry_count + 1}")
            alpha = min(max_alpha, alpha + increment)  # Increment alpha while ensuring it doesn't exceed max_alpha
            retry_count += 1
    raise ValueError("Failed to find a valid alpha within the allowed retries.")

def get_qhat(algo, X_val_fold, y_val_fold, alpha1_start=0.05, alpha2_start=0.1):
    probabilities = algo.predict_proba(X_val_fold)
    
    # Check if log transformation is needed
    if np.max(probabilities) < 0.01:
        log_transform = True
        probabilities = np.log(probabilities + 1e-10)  # Apply log transformation
    else:
        log_transform = False
        
    softmax_outputs_pos = probabilities[:,1]
    softmax_outputs_neg = probabilities[:,0]
    
    N1 = softmax_outputs_pos.shape[0]
    scores1=np.zeros(N1)
    
    counts = np.bincount(y_val_fold)
    print(f"Number of 0s in y_val_fold: {counts[0]}")
    print(f"Number of 1s in y_val_fold: {counts[1]}")

    for i in range(N1):
        if y_val_fold[i] == 1:
            scores1[i]= softmax_outputs_pos[i]
        else:
             scores1[i] = 1-softmax_outputs_neg[i]

        
    non_zero_count1 = np.count_nonzero(scores1)
    non_zero_scores1 = scores1[scores1 != 0]
    
    alpha1 = alpha1_start
    while alpha1 <= 1.0:
        quantile_value1 = np.ceil((counts[1] + 1) * (1 - alpha1)) / counts[1]
        if 0 <= quantile_value1 <= 1:
            break
        alpha1 += 0.01

    val1 = np.quantile(non_zero_scores1, quantile_value1)
    if log_transform:
        val1 = np.exp(val1)  # Back transform if log was applied
            
    val1 = np.quantile(scores1,np.ceil((N1+1)*(1-alpha1))/N1)


    N2 = softmax_outputs_neg.shape[0]
    scores2=np.zeros(N2)
    

    for i in range(N2):
        if y_val_fold[i] == 0:
            scores2[i]= softmax_outputs_neg[i]
        else:
            scores2[i] = 1-softmax_outputs_pos[i]

    non_zero_scores2 = scores2[scores2 != 0]

    alpha2 = alpha2_start
    while alpha2 <= 1.0:
        quantile_value2 = np.ceil((counts[0] + 1) * (1 - alpha2)) / counts[0]
        if 0 <= quantile_value2 <= 1:
            break
        alpha2 += 0.01

    val2 = np.quantile(non_zero_scores2, quantile_value2)
    if log_transform:
        val2 = np.exp(val2)  # Back transform if log was applied

        
    return val1, val2

def get_conformal_set(probabilities, feature, val1, val2 ):
    # probabilities = algo.predict_proba(feature)

    # Split the probabilities for positive and negative classes
    prob_positive_class = probabilities[:, 1]
    prob_negative_class = probabilities[:, 0]

    N = len(feature)

    conf_sets = []
    for i in range(N):
        if prob_positive_class[i] >= val1: #changed self.q_yhat_pos to val
            conf_sets.append('1')  # Predicted as positive
        else:
            conf_sets.append('(0,1)')  # Predicted as not sure
            
    for i in range(N):
        if prob_negative_class[i] >= val2:
            conf_sets[i] = '0' 
            
    return conf_sets


def get_conformal_values(conf_sets):
    conf_values = np.zeros(len(conf_sets))
    for i in range(len(conf_sets)):
        if conf_sets[i] == '1':
            conf_values[i] = 1
        elif conf_sets[i] == '0':
            conf_values[i] = 0
        else:
            conf_values[i] = 0.5
    return conf_values

# Function to calculate accuracy
def binary_accuracy(y_pred, y_true):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_true).float().sum()
    acc = correct_results_sum / y_true.shape[0]
    acc = torch.round(acc * 100)
    return acc

def show_dataset_map(X_train_fold, test_mask_list, index,name):

    train_pos = X_train_fold[:, -2:]
    mask = np.zeros_like(test_mask_list[0])
    for x, y in train_pos:
        mask[int(x), int(y)] = 1

    plt.figure(dpi=600)
    plt.imshow(mask, cmap='viridis')
    plt.rcParams['font.size'] = 18
    file_path = os.path.abspath(__file__)
    file_dir = os.path.dirname(file_path)
    path_save = os.path.join(file_dir, '/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/')

    plt.savefig(path_save+name+'_'+str(index)+'.png', dpi=300)
    plt.close()
    
    return

# Plot Precision-Recall and ROC curves
def plot_precision_recall(y_true, y_scores, index, name):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    path_save = "G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/"

    plt.savefig(path_save+name+'_'+str(index)+'.png', dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_scores, index, name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, marker='.', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    path_save = "G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/"

    plt.savefig(path_save+name+'_'+str(index)+'.png', dpi=300)
    plt.close()

def align_results_with_positions(result_values, positions, mask_shape):
    aligned_result_array = np.zeros(mask_shape, dtype="float")
    for i, pos in enumerate(positions):
        if i < len(result_values):
            aligned_result_array[int(pos[0]), int(pos[1])] = result_values[i]
    return aligned_result_array


def add_mask(test_mask):
    new_mask = np.zeros_like(test_mask[0])
    # 对test_mask[0]、test_mask[1]、test_mask[2]、test_mask[3]进行或操作
    for i in range(len(test_mask)):
        new_mask = new_mask | test_mask[i]
    new_mask = ~new_mask
    test_mask = np.concatenate((test_mask,np.expand_dims(new_mask, axis=0)),axis=0)
    return test_mask



def get_mask(X_train_fold, X_test_fold, test_mask_list):

    train_pos = X_train_fold[:, -2:]
    train_mask = np.zeros_like(test_mask_list[0])
    for x, y in train_pos:
        train_mask[int(x), int(y)] = 1

    test_pos = X_test_fold[:, -2:]
    test_mask = np.zeros_like(test_mask_list[0])
    for x, y in test_pos:
        test_mask[int(x), int(y)] = 1
    all_mask = np.ones_like(test_mask_list[0])
    return train_mask, test_mask

def image_conformal_withdeposit_forfold(result_values, mask, deposit_mask, test_mask=None, index=0, clusters=4, name='result_map', filter=False):
    plt.figure(dpi=600)
    plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=False)

    validYArray, validXArray = np.where(mask > 0)
    # dep_YArray, dep_XArray = np.where(np.logical_and(deposit_mask, test_mask) == 1)
    dep_YArray, dep_XArray = np.where(deposit_mask > 0)
    result_array = np.zeros_like(deposit_mask, dtype="float")

    plt.scatter(dep_XArray, dep_YArray, c='red', s=20, alpha=0.5, label='Deposit Mask')  # Overlay deposit mask
    
    for i in range(len(validYArray)):
        if test_mask[validYArray[i], validXArray[i]] > 0:
            result_array[validYArray[i], validXArray[i]] = result_values[validYArray[i], validXArray[i]]

    if filter:
        result_array = median_filter(result_array, size=3)

    result_array[~mask] = np.nan

    colors = ['darkblue', 'yellow', 'lime']
    labels = ['Very Low Probability of Gold', 'Moderate Probability of Gold', 'Very High Probability of Gold']
    cmap = ListedColormap(colors) 
    
       
    plt.imshow(result_array, cmap=cmap)
    
    patches = [plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=colors[i], 
                        label="{:s}".format(labels[i]))[0] for i in range(len(colors))]
    plt.legend(handles=patches, bbox_to_anchor=(1.0, 0), loc='lower right', borderaxespad=0.)
    
    # Plot only the deposit points that are within the test mask
    if test_mask is not None:
        dep_masked_YArray, dep_masked_XArray = np.where((deposit_mask > 0) & (test_mask > 0))
        plt.scatter(dep_masked_XArray, dep_masked_YArray, c='red', s=40, alpha=0.5, label='Deposit Mask')  
    else:
        plt.scatter(dep_XArray, dep_YArray, c='red', s=40, alpha=0.5, label='Deposit Mask')  


    path_save = "G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/"

    plt.savefig(path_save+name+'_'+str(index)+'.png', dpi=300)
    
def image_conformal_withdeposit(result_values, mask, deposit_mask, test_mask=None, index=0, clusters=4, name='result_map', filter=False):
    plt.figure(dpi=600)
    plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=False)

    validYArray, validXArray = np.where(mask > 0)
    # dep_YArray, dep_XArray = np.where(np.logical_and(deposit_mask, test_mask) == 1)
    dep_YArray, dep_XArray = np.where(deposit_mask > 0)
    result_array = np.zeros_like(deposit_mask, dtype="float")

    plt.scatter(dep_XArray, dep_YArray, c='red', s=20, alpha=0.5, label='Deposit Mask')  # Overlay deposit mask
    
    for i in range(len(validYArray)):
        if test_mask[validYArray[i], validXArray[i]] > 0:
            result_array[validYArray[i], validXArray[i]] = result_values[validYArray[i], validXArray[i]]

    if filter:
        result_array = median_filter(result_array, size=3)

    result_array[~mask] = np.nan

    colors = ['darkblue', 'yellow', 'lime']
    labels = ['Very Low Probability of Gold', 'Moderate Probability of Gold', 'Very High Probability of Gold']
    cmap = ListedColormap(colors) 
    
       
    plt.imshow(result_array, cmap=cmap)

     

    patches = [plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=colors[i], 
                        label="{:s}".format(labels[i]))[0] for i in range(len(colors))]
    plt.legend(handles=patches, bbox_to_anchor=(1.0, 0), loc='lower right', borderaxespad=0.)


    # Save the figure
    path_save = "G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/"

    plt.savefig(path_save+name+'_'+str(index)+'.png', dpi=300)
    
  
def show_result_map(result_values, mask, deposit_mask, test_mask=None, index=0, clusters=4, name='result_map', filter=False):
    plt.figure(dpi=600)
    plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=False)

    validYArray, validXArray = np.where(mask > 0)
    dep_YArray, dep_XArray = np.where(np.logical_and(deposit_mask, test_mask) == 1)
    result_array = np.zeros_like(deposit_mask, dtype="float")
    
    plt.scatter(dep_XArray, dep_YArray, c='red', s=20, alpha=0.5, label='Deposit Mask')  # Overlay deposit mask

    min_length = min(len(validYArray), len(result_values))

    for i in range(len(validYArray)):
        if test_mask[validYArray[i], validXArray[i]] > 0:
            result_array[validYArray[i], validXArray[i]] = result_values[validYArray[i], validXArray[i]] * 100

    if filter:
        result_array = median_filter(result_array, size=3)

    result_array[~mask] = np.nan

    plt.imshow(result_array, cmap='viridis')
    plt.rcParams['font.size'] = 18

    cbar = plt.colorbar(shrink=0.75, aspect=30, pad=0.06)
    cbar.ax.set_ylabel('Prediction', fontsize=25)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    plt.gca().set_facecolor('lightgray')

    path_save = "G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/"
    plt.savefig(path_save + name + '_' + str(index) + '.png', dpi=300)
    plt.close()  # Close the figure to free memory

class Model:
    DEFAULT_METRIC = roc_auc_score
    DEFAULT_FIDELITY = 5
    DEFAULT_TEST_CLUSTERS = 4
    WARNING_FIDELITY_HIGH = 20
    
    def __init__(self, data_path, data_path2, fidelity=1, test_clusters=4, algorithm=rfcAlgo, mode='random', modify=False):
        with open(data_path, 'rb') as f:
            feature_arr, label_arr, common_mask, feature_name_list = pickle.load(f)
        self.set_fidelity(fidelity)
        self.set_test_clusters(test_clusters) 
        self.height, self.width = common_mask.shape
        self.label_arr = label_arr.astype(int)
        self.feature_name_list = feature_name_list
        
        with open(data_path2, 'rb') as f:
            _, _, common_mask_old, deposite_mask_old = pickle.load(f)

        new_shape = (self.height, self.width)
        self.common_mask_new = np.zeros(new_shape, dtype=common_mask_old.dtype)
        self.common_mask_new[:common_mask_old.shape[0], :common_mask_old.shape[1]] = common_mask_old

        self.deposit_mask_new = np.zeros(new_shape, dtype=deposite_mask_old.dtype)
        self.deposit_mask_new[:deposite_mask_old.shape[0], :deposite_mask_old.shape[1]] = deposite_mask_old
        
        scaler = StandardScaler()
        feature_arr[:,2:][feature_arr[:,2:]<1e-5] = 0
        self.feature_arr = scaler.fit_transform(feature_arr)

        x_arr, y_arr = common_mask.nonzero()
        positive_x = x_arr[self.label_arr.astype(bool)].reshape((-1,1))
        positive_y = y_arr[self.label_arr.astype(bool)].reshape((-1,1))

        row_idx, col_idx = np.where(common_mask)

        # 然后使用 row_idx 和 col_idx 来构建 label_arr2d
        self.label_arr2d = np.zeros(common_mask.shape, dtype=int)
        self.label_arr2d[row_idx, col_idx] = ~self.label_arr

        self.real_mask = common_mask
        self.common_mask = np.ones_like(self.real_mask)

        self.algorithm = algorithm
        self.path = data_path
        self.mode = mode
        self.modify = modify
        self.pred_list = None

        return


    def set_test_clusters(self, test_clusters=DEFAULT_TEST_CLUSTERS):
        if not isinstance(test_clusters, int):
            raise RuntimeError("The test_clusters must be an integer!")
        if test_clusters <= 1:
            raise RuntimeError(f"The test_clusters must be more than 1, but now it is {test_clusters}!")
            
        self.test_clusters = test_clusters
        
    def set_fidelity(self, fidelity=DEFAULT_FIDELITY):
        if not isinstance(fidelity, int):
            raise RuntimeError("The fidelity must be an integer!")
        if fidelity < 1:
            raise RuntimeError(f"The fidelity must be positive, but now it is {fidelity}!")
        if fidelity > Model.WARNING_FIDELITY_HIGH:
            warnings.warn(f"The fidelity is suspiciously high. It is {fidelity}.")
            
        self.fidelity = fidelity
        
    def km(self, x, y, cluster):

        coord = np.concatenate([x, y], axis=1)
        cl = KMeans(n_clusters=cluster, random_state=0).fit(coord)
        cll = cl.labels_
        
        return cll
    


    def test_extend(self, x, y, test_num):

        # Build the test mask
        test_mask = np.zeros_like(self.common_mask).astype(bool)
        test_mask[x, y] = True
        
        common_mask = self.common_mask

        candidate = set([])
        for i in range(test_num-1):
            # Add the neighbor grid which is in the valid region and not chosen yet into the candidate set
            if x >= 1 and common_mask[x-1, y] and not test_mask[x-1, y]:
                candidate.add((x-1, y))
            if y >= 1 and common_mask[x, y-1] and not test_mask[x, y-1]:
                candidate.add((x, y-1))
            if x <= self.height-2 and common_mask[x+1, y] and not test_mask[x+1, y]:
                candidate.add((x+1, y))
            if y <= self.width-2 and common_mask[x, y+1] and not test_mask[x, y+1]:
                candidate.add((x, y+1))
            
            # Randomly choose the next grid to put in the test set
            try:
                pick = np.random.randint(0, len(candidate))
            except:
                test_mask[x, y] = True
                continue
            x, y = list(candidate)[pick]
            candidate.remove((x,y))
            test_mask[x, y] = True
            
        return test_mask
    
    def dataset_split(self, test_mask_list=None, modify=True):
    
        if test_mask_list is None:
            test_mask_list = []
            # Randomly choose the start grid
            
            mask_sum = self.real_mask.sum()
            test_num = int(mask_sum / self.test_clusters)
            x_arr, y_arr = self.real_mask.nonzero()
            positive_x = x_arr[self.label_arr.astype(bool)].reshape((-1,1))
            positive_y = y_arr[self.label_arr.astype(bool)].reshape((-1,1))
            cll = self.km(positive_x, positive_y, self.test_clusters)
            
            for i in range(self.test_clusters):
                cluster_arr = (cll == i)
                cluster_x = positive_x[cluster_arr].squeeze()
                cluster_y = positive_y[cluster_arr].squeeze()
                
                start = np.random.randint(0, cluster_arr.sum())
                x, y = cluster_x[start], cluster_y[start]
                test_mask = self.test_extend(x, y, test_num)
                test_mask_list.append(test_mask)
        else:
            for test_mask in test_mask_list:
                assert test_mask.shape == self.common_mask.shape
        # Buf the test mask
        tmpt = test_mask_list
        # Split the dataset
        dataset_list = []
        for test_mask in test_mask_list:
            train_mask = ~test_mask
            
            test_mask = test_mask & self.real_mask
            test_pos = np.where(test_mask)
            test_pos = np.array(test_pos)
            
            test_mask = test_mask[self.real_mask]
            
            train_mask = train_mask & self.real_mask
            
            mask_sum = train_mask.sum()
            val_num = int(mask_sum / 4)
            x_arr, y_arr = train_mask.nonzero()
            train_deposite_mask = train_mask & self.label_arr2d
            
            
            positive_indices = np.argwhere(train_deposite_mask)
            positive_x = np.array([positive_indices[:, 0]]).reshape(-1, 1)
            positive_y = np.array([positive_indices[:, 1]]).reshape(-1, 1)
            
            cll = self.km(positive_x, positive_y, self.test_clusters)
            
            retries = 0
            while retries < 1:
                cluster_arr = (cll == 0)
                cluster_x = np.atleast_1d(positive_x[cluster_arr].squeeze())
                cluster_y = np.atleast_1d(positive_y[cluster_arr].squeeze())

                # Check if the cluster is empty
                if cluster_x.size > 0 and cluster_y.size > 0:
                    start = np.random.randint(0, cluster_arr.sum())
                    x, y = cluster_x[start], cluster_y[start]
                    val_mask = self.test_extend(x, y, val_num)
                    train_mask = train_mask & ~val_mask
                    
                    val_pos = np.where(val_mask)
                    val_pos = np.array(val_pos)
                    
                    val_mask = val_mask[self.real_mask]
                    X_val_fold, y_val_fold = self.feature_arr[val_mask], self.label_arr[val_mask]
                    break  # Exit the while loop if a valid cluster is found
                else:
                    retries += 1
                    print(f"Validation cluster is empty. Retrying... ({retries}/{3})")
            if retries == 3:
                print("Failed to find a non-empty validation cluster after {max_retries} retries. Skipping...")
            
            
            train_pos = np.where(train_mask)
            train_pos = np.array(train_pos)
            
            train_mask = train_mask[self.real_mask]
            
            X_train_fold, X_test_fold = self.feature_arr[train_mask], self.feature_arr[test_mask]
            y_train_fold, y_test_fold = self.label_arr[train_mask], self.label_arr[test_mask]
            
    
            test_pos = np.transpose(test_pos)
            X_test_fold = np.concatenate([X_test_fold, test_pos], axis=1)
            train_pos = np.transpose(train_pos)
            X_train_fold = np.concatenate([X_train_fold, train_pos], axis=1)
                
            if modify:
                true_num = y_train_fold.sum()
                index = np.arange(len(y_train_fold))
                true_train = index[y_train_fold == 1]
                false_train = np.random.permutation(index[y_train_fold == 0])[:true_num]
                train = np.concatenate([true_train, false_train])
                X_train_fold = X_train_fold[train]
                y_train_fold = y_train_fold[train]
                



            X_train_fold, y_train_fold = sk_shuffle(X_train_fold, y_train_fold)
                
            dataset = (X_train_fold, y_train_fold, X_val_fold, y_val_fold,  X_test_fold, y_test_fold)
            dataset_list.append(dataset)
        
        return tmpt, dataset_list
            

    
    def random_spilt(self, modify =False):

        feature = self.feature_arr

        pos = np.where(self.common_mask)
        pos = np.array(pos)
        pos = np.transpose(pos)

        feature = np.concatenate([feature, pos], axis=1)

        total_label = self.total_label_arr
        # print("Total positive sum", total_label.sum())
        ground_label = total_label[0]
        aug_label = total_label[1]
        
        dataset_list = []
        kf = KFold(n_splits=5, shuffle=True)
        for train_index , test_index in kf.split(feature): 
            
            X_train_fold, X_test_fold, y_train_fold, y_test_fold = [],[],[],[]
            X_val_fold, y_val_fold = [], []
            for i in train_index:
                X_train_fold.append(feature[i])
                y_train_fold.append(aug_label[i])

            # split the val set
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
                                                                    X_train_fold, 
                                                                    y_train_fold, 
                                                                    test_size=0.25)
            
            for i in test_index:
                X_test_fold.append(feature[i])
                y_test_fold.append(ground_label[i])

            X_train_fold, X_test_fold, y_train_fold, y_test_fold = np.array(X_train_fold), np.array(X_test_fold), np.array(y_train_fold), np.array(y_test_fold)
            X_val_fold, y_val_fold = np.array(X_val_fold), np.array(y_val_fold)

            if y_test_fold.sum() == 0: 
                continue

            if modify:
                true_num = y_train_fold.sum()
                index = np.arange(len(y_train_fold))
                true_train = index[y_train_fold == 1]
                false_train = np.random.permutation(index[y_train_fold == 0])[:true_num]
                train = np.concatenate([true_train, false_train])
                X_train_fold = X_train_fold[train]
                y_train_fold = y_train_fold[train]

            dataset = (X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_fold, y_test_fold)
            dataset_list.append(dataset)
            
        return dataset_list
    
    
    def prediction_intervals(self, probs, confidence_level=0.95):
        
        all_lower_bounds = []
        all_upper_bounds = []
        z = 1.96  # z-value for 95% confidence interval

        interval_width = z * np.sqrt(probs * (1 - probs))
        lower_bound = np.clip(probs - interval_width, 0, 1)  # Ensure bounds are within [0, 1]
        upper_bound = np.clip(probs + interval_width, 0, 1)  # Ensure bounds are within [0, 1]

        all_lower_bounds.extend(lower_bound)
        all_upper_bounds.extend(upper_bound)
        
        lower_bounds = np.array(all_lower_bounds)
        upper_bounds = np.array(all_upper_bounds)
        return np.column_stack((lower_bounds, upper_bounds))

    def coverage_width(self, prediction_intervals):
        lower_bounds, upper_bounds = prediction_intervals[:, 0], prediction_intervals[:, 1]
        coverage_widths = upper_bounds - lower_bounds
        average_coverage_width = np.mean(coverage_widths)
        return average_coverage_width
        
            
    def train(self, params,  metrics=['roc_auc_score', 'f1_score', 'precision_score', 'recall_score'], test_mask=None, modify=False):
        
        modify = self.modify
        if not isinstance(metrics, list):
            metrics = [metrics]
        metric_list = []

        for metric in metrics:
            if isinstance(metric, str):
                if metric.lower() == 'roc_auc_score' or metric.lower() == 'auc' or metric.lower() == 'auroc':
                    metric_func = roc_auc_score
                elif metric.lower() == 'f1_score' or metric.lower() == 'f1':
                    metric_func = f1_score
                elif metric.lower() == 'precision_score' or metric.lower() == 'pre':
                    metric_func = precision_score
                elif metric.lower() == 'recall_score' or metric.lower() == 'recall':
                    metric_func = recall_score
                else:
                    warnings.warn(f'Wrong metric! Replace it with default metric {Model.DEFAULT_METRIC.__name__}.')
                    metric_func = Model.DEFAULT_METRIC
            elif isfunction(metric):
                metric_func = metric
            else:
                warnings.warn(f'Wrong metric! Replace it with default metric {Model.DEFAULT_METRIC.__name__}.')
                metric_func = Model.DEFAULT_METRIC
            metric_list.append(metric_func)
            
        score_list, cfm_list, pred_list = [], [], []
        X_fold_sum, shap_values_list = [], []

        if self.mode  == 'IID':
            print("Training with IID mode")
            dataset_list = self.random_spilt(modify=modify)
            i = 0 
            y_arr_record = []
            for dataset in dataset_list:
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_fold, y_test_fold = dataset

                # 可视化训练集
                show_dataset_map(X_train_fold,[self.common_mask],i+1,'train')
                # 可视化测试集
                show_dataset_map(X_test_fold,[self.common_mask],i+1,'test')

                algo = self.algorithm(params)
                algo.fit(X_train_fold[:,:-2], y_train_fold)
                
                scores = []
                pred_arr, y_arr = algo.predictor(X_test_fold[:,:-2])
                # set the optimization target
                scores.append(criterion_loss(self.mode, algo, X_val_fold[:,:-2], y_val_fold, y_train_fold))

                # test the performance
                for metric in metric_list:
                    if metric != roc_auc_score:
                        score = metric(y_true=y_test_fold, y_pred=pred_arr)
                        scores.append(score) 
                    else:
                        score = metric(y_test_fold, y_arr)
                        scores.append(score)
                        print("ROC AUC score", score)

                if len(scores) == 1:
                    scores = scores[0]
                score_list.append(scores)
                _, y_arr_feature_map = algo.predictor(self.feature_arr)
                y_arr_record.append(y_arr_feature_map)

                _train_mask, _test_mask= get_mask(X_train_fold, X_test_fold, [self.deposit_mask])

                show_result_map(result_values=y_arr_record[i], mask=self.common_mask, deposit_mask=self.deposit_mask, test_mask=self.common_mask, index=i+1, name='all_pred')
                show_result_map(result_values=y_arr_record[i], mask=self.common_mask, deposit_mask=self.deposit_mask, test_mask=_train_mask, index=i+1, name='train_pred')
                show_result_map(result_values=y_arr_record[i], mask=self.common_mask, deposit_mask=self.deposit_mask, test_mask=_test_mask, index=i+1, name='test_pred')
                
                
                test_mask_list = np.zeros_like(self.common_mask)
                for j in range(len(X_test_fold)):
                    x, y = X_test_fold[j][-2:]
                    test_mask_list[int(x), int(y)] = 1

                y_mask = test_mask_list.reshape(-1)

                y_mask = y_mask[self.common_mask.reshape(-1)]
                y_arr_record[i][~y_mask] = 0

                i += 1
            y_arr_record = np.maximum.reduce(y_arr_record)
            show_result_map(result_values=y_arr_record, mask=self.common_mask, deposit_mask=self.deposit_mask, test_mask=self.common_mask, index=0, name='concatenate_all_pred',filter=False)


            
        else: 
            
            print("Training with OOD mode")
            #test_mask = np.load('./Bayesian_main/temp/New_Nova_mask.npy')

            # To get a complete prediction plot, add a mask
            #test_mask = add_mask(test_mask)

            test_mask_list, dataset_list = self.dataset_split()
            
            tpr_list = []
            roc_list = []
            f1_list = []
            fpr_list = []
            y_arr_record = []
            i = 0
            
            all_pos = np.where(self.real_mask)
            all_pos = np.array(all_pos)
            all_pos = np.transpose(all_pos)
            new_feature_arr = np.concatenate([self.feature_arr, all_pos], axis=1)
            test_outputs_total = []
            test_labels_total = []
            uncertainties = []
            coverage = []

            for dataset in dataset_list:
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_fold, y_test_fold = dataset
                
                print("X_train_fold", X_train_fold[:,:-2].shape)
                print("X_val_fold", X_val_fold.shape)
                print("X_test_fold", X_test_fold[:,:-2].shape)

                # show_dataset_map(X_train_fold,test_mask_list,i+1,'train')
                show_dataset_map(X_test_fold,test_mask_list,i+1,'test')

                # _train_mask, _test_mask= get_mask(X_train_fold, X_test_fold, test_mask_list)

                algo = self.algorithm(params)
                algo.fit(X_train_fold[:, :-2], y_train_fold)
                scores = []
                pred_arr, y_arr = algo.predictor(X_test_fold[:, :-2])
                
                # set the optimization target
                scores.append(criterion_loss(self.mode, algo, X_val_fold, y_val_fold))
                
                test_outputs_total.extend(y_arr)
                test_labels_total.extend(y_test_fold)

                
                for metric in metric_list:
                    if metric != roc_auc_score:
                        score = metric(y_true=y_test_fold, y_pred=pred_arr)
                        scores.append(score)
                    else:
                        score = metric(y_test_fold, y_arr)
                        scores.append(score)  

                plot_precision_recall(y_test_fold, y_arr, index=i+1, name='PR_curve')
                plot_roc_curve(y_test_fold, y_arr, index=i+1, name='roc_curve')

                if len(scores) == 1:
                    scores = scores[0]
                score_list.append(scores)
                
                print(f'Test AUC: {scores[1]:.4f}, Test F1: {scores[2]:.4f}, precission: {scores[3]:.4f}, recall: {scores[4]:.4f}')
    
                uncertainty_values = np.var(pred_arr, axis=0)
                # Print or save the predictive uncertainty values
                print("Predictive Uncertainty for OOD:", uncertainty_values)
                
                uncertainties.append(uncertainty_values)
               
                prediction_intervals = self.prediction_intervals(y_arr)
                average_coverage_width = self.coverage_width(prediction_intervals)
                print("coverage_width", average_coverage_width)
                coverage.append(average_coverage_width)
    
                tn, fp, fn, tp = confusion_matrix(y_test_fold, pred_arr).ravel()
                tpr = tp / (tp + fn)
                tpr_list.append(tpr)
                fpr = fp / (fp + tn)
                fpr_list.append(fpr)
                print(f'Test TPR: {tpr:.4f}, Test FPR: {fpr:.4f}')         
                roc = scores[1]
                roc_list.append(roc)
                f1 = scores[2]
                f1_list.append(f1)
                
                # Assuming the last two columns of X_test_fold are the positions
                test_positions = X_test_fold[:, -2:]
                
                # Generate test mask for visualization
                test_mask = np.zeros_like(self.real_mask, dtype=bool)
                for pos in test_positions:
                    test_mask[int(pos[0]), int(pos[1])] = True
                    
                all_positions = new_feature_arr[:, -2:]
    
                # Generate test mask for visualization
                all_mask = np.zeros_like(self.real_mask, dtype=bool)
                for pos in all_positions:
                    all_mask[int(pos[0]), int(pos[1])] = True
                

                _, y_arr_feature_map = algo.predictor(new_feature_arr[:,:-2])
                  
    
                aligned_full_results = align_results_with_positions(y_arr_feature_map, all_positions, self.real_mask.shape)
                y_arr_record.append(aligned_full_results)
                
                
                
                # show_result_map(result_values=aligned_full_results, mask=self.common_mask_new, deposit_mask=self.deposit_mask_new, test_mask=_test_mask, index=i+1, name='test_pred')
                # show_result_map(result_values=aligned_full_results, mask=self.common_mask_new, deposit_mask=self.deposit_mask_new, test_mask=self.common_mask_new, index=i+1, name='all_pred')

            
                
                y_mask = test_mask_list[i].reshape(-1)
                y_mask = y_mask[self.common_mask.reshape(-1)]
                y_mask = y_mask.reshape(self.common_mask_new.shape)
                y_arr_record[i][~y_mask] = 0
      
                i += 1

            
                        
            test_auc = roc_auc_score(test_labels_total, test_outputs_total)
            test_f1 = f1_score(test_labels_total, np.round(test_outputs_total))
                        
            print(f'total test AUC: {test_auc:.4f}, test F1: {test_f1:.4f}')
                
            tn, fp, fn, tp = confusion_matrix(test_labels_total, np.round(test_outputs_total)).ravel()
            tpr = tp / (tp + fn)  # True Positive Rate
            fpr = fp / (fp + tn)  # False Positive Rate

            print(f'Test TPR: {tpr:.4f}, Test FPR: {fpr:.4f}')
                
            # plot_precision_recall(test_labels_total, test_outputs_total, index=6, name='PR_curve')
            # plot_roc_curve(test_labels_total, test_outputs_total, index=6, name='roc_curve')           
            # Save the model after training
            y_arr_record = np.maximum.reduce(y_arr_record)
            # show_result_map(result_values=y_arr_record, mask=self.common_mask_new, deposit_mask=self.deposit_mask_new, test_mask=self.common_mask_new, index=0, name='concatenate_all_pred',filter=True)

            roc2 = sum(roc_list)/len(roc_list)
            f1_score2 = sum(f1_list)/len(f1_list)
            fpr2 = sum(fpr_list)/len(fpr_list)
            print(roc2)
            print(f1_score2)
            print(fpr2)
            
            PU = sum(uncertainties)/len(uncertainties)
            CW = sum(coverage)/len(coverage)
            
            print(PU)
            print(CW)


        return score_list
    
    def obj_train_parallel(self, queue, args):

        score = self.train(args)
        queue.put(score)
        return
    
    def evaluate(self, x):

        queue = Queue()
        process_list = []
        for _ in range(self.fidelity):
            p = Process(target=self.obj_train_parallel, args=[queue, x])
            p.start()
            process_list.append(p)
            
        for p in process_list:
            p.join()
            
        score_list = []
        for i in range(self.fidelity):
            score_list.append(queue.get())

        y = np.mean(
            np.mean(score_list, axis=0),
            axis=0
            )
        std_arr = np.std(score_list, axis=1)
        auc_std, f_std = np.mean(std_arr[:,1]), np.mean(std_arr[:,2])
        y = list(y) + [auc_std, f_std]
        
        
        return y

        

# For debugging
if __name__ == '__main__':

    # fix the random seed
    np.random.seed(1)
    algo = rfcAlgo
    # algo = rfBoostAlgo
    # algo = svmAlgo
    if algo == svmAlgo:
        x = {'C':0.9, 'kernel':'rbf', 'probability': True}
    elif algo == logiAlgo:
        x = {'penalty':'l2', 'max_iter':1000, 'solver':'saga'}
    elif algo == extAlgo:
        x = {'n_estimators': 30, 'max_depth':15}
    elif algo == rfcAlgo:
        x = {'n_estimators': 10, 'max_depth':5}

    else:
        x = {'n_estimators': 50}
    print(x)

    task = Model(
        data_path='G:/A PYTHON NOTEBOOK/Bayesian_main/ooddata/Washington_new.pkl',
        data_path2 = 'G:/A PYTHON NOTEBOOK/Bayesian_main/ooddata/Washington.pkl',
        algorithm=algo,
        mode='OOD',
        modify=True,
        test_clusters=4
        )
    
    y = task.evaluate(x)
    print(f'{algo.__name__}: {y}')