from scipy import stats
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import norm
import math
from sklearn.linear_model import SGDRegressor
from nonconformist.icp import IcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc
from Bayesian_main.code.utils3 import preprocess_data, preprocess_Nova_data
import pickle
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score
from matplotlib.colors import ListedColormap


def image_conformal(reg_mask,result_values,dep_mask):
    validYArray, validXArray = np.where(reg_mask > 0)
    result_array = np.zeros_like(dep_mask, dtype="float")
    for i in range(len(validYArray)):
        if result_values[i] is None:    
            result_array[validYArray[i], validXArray[i]] = -1
        elif result_values[i] == (0,1):
            result_array[validYArray[i], validXArray[i]] = 0.5
        elif result_values[i] == 0:
            result_array[validYArray[i], validXArray[i]] = 0
        else:
            result_array[validYArray[i], validXArray[i]] = 1
    
    #print("There's something: ",validXArray,validYArray)
    result_array[~reg_mask] = np.nan
    print(result_array)
    colors = ['purple', 'blue', 'red', 'green', 'lightgray']
    cmap = ListedColormap(colors)
    plt.imshow(result_array, cmap=cmap)
    
    plt.rcParams['font.size'] = 18


def get_csv(x, path, directory, list = None):
    column_names = ["背斜缓冲区","背斜缓冲区", "砷", "锂", "铅", "氟", "铜", "钨", "锌"]
    
    df = pd.DataFrame(x, columns=column_names)
    if list is None:
        df["输出"] = "-"
    else:
        df["输出"] = list
    try:
    # Create the directory
        os.mkdir(directory)
        print(f"Directory '{directory}' created.")
    except FileExistsError:
        print(f"Directory '{directory}' already exists.")
        
    file_path = os.path.abspath(os.path.join(directory, path))
    with open(file_path, 'w') as f:
        pass
    df.to_csv(file_path, index = False)
    return df


def kfold_cv_split(X, y, n_splits=5, random_state=22):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_cal, X_val, y_cal, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=random_state)

        yield X_train, y_train, X_cal, y_cal, X_val, y_val

def train_cal_test_split(X,y,random_state=22):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=random_state)
    X_cal,X_val,y_cal,y_val= train_test_split(X_test, y_test, test_size=0.5, random_state=random_state)
    return X_train,y_train,X_cal,y_cal,X_val,y_val



class Conformal:
    """
    Conformal prediction for the dataset, using the model and hyperparameters taken from BO, will be coupled with BO to yield
    an output with the classes already.
    """
    def __init__(self, datapath, model, dataset = None, alpha = 0.1):
        self.dataset = dataset
        self.data_p = datapath
        self.model = model
        self.alpha = alpha
        self.X= None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_cal = None
        self.y_cal = None
        self.X_val = None
        self.y_val = None
        #self.q_yhat_pos = 0
        #self.q_yhat_neg = 0
        #self.conf_sets_pos = []
        #self.conf_sets_neg = []
        self.coverage_pos = 0
        self.coverage_neg = 0
        self.feature_matrix = []
        self.deposit = None
        self.mask = None
    
    def get_data_from_p(self):
        with open(self.data_p, 'rb+') as file:
    
            data = pickle.load(file)
        X = data[0]
        y = data[1][1]
        self.mask = data[2]
        self.deposit = data[3]
        self.X,self.y = X,y
        
    def get_qhat_positive(self):
        #self.get_data_from_p()
        #X_train,y_train,X_cal,y_cal,self.X_val,self.y_val= train_cal_test_split(self.X,self.y)
        self.model.fit(self.X_train,self.y_train)
        softmax_outputs= self.model.predict_proba(self.X_cal)[:,1]
        N=softmax_outputs.shape[0]
        scores=np.zeros(N)
    
        for i in range(N):
            if self.y_cal[i] == 1:
                scores[i]= 1-softmax_outputs[i]
            else:
                scores[i] = softmax_outputs[i]

        #self.q_yhat_pos = np.quantile(scores,np.ceil((N+1)*(1-self.alpha))/N)
        return np.quantile(scores,np.ceil((N+1)*(1-self.alpha))/N)
        
    def get_qhat_negative(self):
        #X,y = self.get_data_from_p()
        #X_train,y_train,X_cal,y_cal,self.X_val,self.y_val= train_cal_test_split(X,y)
        self.model.fit(self.X_train,self.y_train)
        softmax_outputs= self.model.predict_proba(self.X_cal)[:,0] # get the probability of negative
        #print("PROBS for neg: ", softmax_outputs)
        N=softmax_outputs.shape[0]
        scores=np.zeros(N)
    
        for i in range(N):
            if self.y_cal[i] == 0:
                scores[i]= 1-softmax_outputs[i]
            else:
                scores[i] = softmax_outputs[i]

        #self.q_yhat_neg = np.quantile(scores,np.ceil((N+1)*(1-self.alpha))/N)
        return np.quantile(scores,np.ceil((N+1)*(1-self.alpha))/N)
    
    def get_confsets_naive_pos(self, val = None):
        # prob_positive_class = self.model.predict(self.X_val)
        prob_positive_class= self.model.predict_proba(self.X_val)[:,1]
        #print(prob_positive_class)
        N = len(self.X_val)

        conf_sets = []

        for i in range(N):
            if prob_positive_class[i] >= 1 - val: #changed self.q_yhat_pos to val
                conf_sets.append((1))  # Predicted as positive
            else:
                
                conf_sets.append((0,1))  # Predicted as not sure

        self.conf_sets_pos = conf_sets
        
    def get_confsets_naive_neg(self, val = None):
        # prob_positive_class = self.model.predict(self.X_val)
        prob_positive_class= self.model.predict_proba(self.X_val)[:,0]
        print(prob_positive_class)
        N = len(self.X_val)

        conf_sets = []

        for i in range(N):
            if prob_positive_class[i] >= 1 - val: # same as in get_confsets_naive_pos
                conf_sets.append(0)  # Predicted as negative
            else:
                
                conf_sets.append((0,1))  # Predicted as not sure

        self.conf_sets_neg = conf_sets
        
    def get_coverage(self):
        total_data_points = len(self.y_val)
        conformal_data_points_pos = 0
        conformal_data_points_neg = 0
    
        for i in range(total_data_points):
        # Check if the true label is in the prediction set (interval)
            if self.y_val[i] == self.conf_sets_pos[i].any():
                conformal_data_points_pos += 1
        
        for i in range(total_data_points):
        # Check if the true label is in the prediction set (interval)
            if self.y_val[i] == self.conf_sets_neg[i].any():
                conformal_data_points_neg += 1

    # Calculate coverage as the proportion of conformal data points
        self.coverage_pos = conformal_data_points_pos / total_data_points
        self.coverage_neg = conformal_data_points_neg / total_data_points
    def positive_ranges(self, silent = True):
        positive = []
        for i in range(len(self.conf_sets)):
            if self.conf_sets[i] == 1:
                positive.append(self.X_val[i])
        if not silent:
            print("This is positive and its length:" ,len(positive), " ----- ", positive )
        n = len(positive[0])
        feature_matrix = [[] for _ in range(n)]
        for i in range(n):
            for j in range(len(positive)):
                feature_matrix[i].append(positive[j][i])
        min_list = [min(feature_matrix[i]) for i in range(len(feature_matrix))]
        max_list = [max(feature_matrix[i]) for i in range(len(feature_matrix))]
        self.feature_matrix = feature_matrix
        return min_list, max_list
    
    def plot_heatmap(self):
        plt.figure(figsize=(10, 6))  # Set the figure size
        plt.imshow(self.feature_matrix, cmap='viridis', aspect='auto')  # Use 'viridis' colormap (you can choose a different colormap)
        plt.colorbar()  # Add a colorbar to show the color-to-value mapping

        # Add labels to the axes (optional)
        plt.xlabel('Samples')
        plt.ylabel('Features')

        # Add a title (optional)
        plt.title('Feature Matrix Heatmap')

        # Show the plot
        plt.show()
    
    def calculate_acc(self):
        accuracy = accuracy_score(self.y_val,self.conf_sets)
        return accuracy
    def calculate_f1(self):
        f1_sc = f1_score(self.y_val,self.conf_sets)
        return f1_sc
    def conformal_image(self, df = None):
        """_summary_
        Here we're going to give a what we call the conformal image of the test data, which has 3 color values
            - Green: There's a mineral
            - Blue: Not sure
            - Red: No mineral for sure
        For that we need the conformal values from the negative and positive, so we can create the image
        self.conf_sets_neg
        self.conf_sets_pos
        """
        
        idx_val = []
        for elem in self.X_val:
            index_array = next((index for index, row in enumerate(self.X) if np.all(elem == row)), None)
            idx_val.append(index_array)
# Now val_indices contains the indices of the validation set in the original X and y

        #print(len(idx_val), type(idx_val), idx_val)
        output = [None]*len(self.X)
        conf_neg = self.conf_sets_neg
        conf_pos = self.conf_sets_pos
       # print("CONF POSITIVE: ", len(conf_pos), type(conf_pos[0]), conf_pos)
       # print("CONF NEGATIVE: ", len(conf_neg), type(conf_neg[0]), conf_neg)
        for i in range(len(conf_pos)):
            if conf_pos[i] == 1:
                output[idx_val[i]] = (1)
            elif conf_neg[i] == 0:
                output[idx_val[i]] = (0)
            else:
                output[idx_val[i]] = (0,1)
                
        #df["输出"] = output
        return output
    
    def conf_pipeline(self, n):
        self.get_data_from_p()
        lists = []
        for fold, (X_train, y_train, X_cal, y_cal, X_val, y_val) in enumerate(kfold_cv_split(self.X, self.y, n_splits=n, random_state=22), 1):
            self.X_train = X_train
            self.y_train = y_train
            self.X_cal = X_cal
            self.y_cal = y_cal
            self.X_val = X_val
            self.y_val = y_val
            qhat_pos = self.get_qhat_positive()
            qhat_neg = self.get_qhat_negative()
            self.get_confsets_naive_pos(qhat_pos)
            self.get_confsets_naive_neg(qhat_neg)
            list = self.conformal_image()
            df = get_csv(self.X, f'data{fold}.csv',f'./csv{fold}', list)
            list_out = df["输出"].tolist()
            lists.append(list_out)
            
            #return list_out
            #image_conformal(self.mask,list_out,self.deposit)
        print(lists)
        return lists
    
    