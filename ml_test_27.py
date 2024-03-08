import pickle
import warnings
from inspect import isfunction
from multiprocessing import Process, Queue
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numpy as np
from algo import *
from sklearn.decomposition import PCA
from sklearn.utils import shuffle as sk_shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, confusion_matrix, recall_score, make_scorer, accuracy_score, average_precision_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import shap
import matplotlib.pyplot as plt
from utils import *
from nonconformist.nc import AbsErrorErrFunc, RegressorNc
from nonconformist.cp import IcpRegressor
from sklearn.ensemble import RandomForestClassifier, IsolationForest, RandomForestRegressor
from econml.dml import CausalForestDML
import pandas as pd
# from metric import Feature_Filter
import seaborn as sns
import sys

warnings.filterwarnings("ignore")


class Model:
    DEFAULT_METRIC = roc_auc_score
    DEFAULT_FIDELITY = 5
    DEFAULT_TEST_CLUSTERS = 5
    WARNING_FIDELITY_HIGH = 20
    
    def __init__(self, data_path, fidelity=1, test_clusters=4, algorithm=rfcAlgo, mode='random', modify=False):
        """
            This file was the model of original maching learning method
            Used for debugging the model of auto-ml

            Nova Scotia : Dataset for testing model performance

            Most of the file is the same to model.py and the explanation of functions involved can 
            be reached in that file
        """
        with open(data_path, 'rb') as f:
            feature_arr, total_label, common_mask, deposite_mask = pickle.load(f)

        self.set_fidelity(fidelity)
        self.set_test_clusters(test_clusters)
        self.feature_arr = feature_arr
        self.feature_arr[:,2:][self.feature_arr[:,2:]<1e-5] = 0
        self.total_label_arr = total_label.astype(int)
        self.label_arr = self.total_label_arr[0]
        self.common_mask = common_mask
        self.deposit_mask = deposite_mask
        self.height, self.width = common_mask.shape
        self.algorithm = algorithm
        self.path = data_path
        self.mode = mode
        self.modify = modify
        self.pred_list = None

        return

    def calculate_correlation_matrix(self):
        # Convert feature array to DataFrame for easier manipulation
        # print("self.feature_arr", self.feature_arr.shape)
        df_features = pd.DataFrame(self.feature_arr)
        # print("df_features", df_features)
        
        # Calculate the correlation matrix
        correlation_matrix = df_features.corr()
        
        return correlation_matrix
    
    def calculate_feature_target_correlation(self):
        # Assumes that the last column of feature_arr is the target variable
        # If not, you need to adjust the code to correctly reference your target variable
        df_features = pd.DataFrame(self.feature_arr)
        target = self.label_arr  # Adjust this if your target variable is located elsewhere

        # Calculate the correlation of each feature with the target variable
        feature_target_correlation = df_features.corrwith(pd.Series(target))

        return feature_target_correlation
    
    def visualize_correlation_matrix(self, correlation_matrix):
    # Using seaborn's heatmap to visualize the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Feature Correlation Matrix")
        plt.savefig("G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/feature_correlation_matrix.png")
        plt.close()
        
    def visualize_feature_target_correlation(self, feature_target_correlation):
        # Visualizing the feature to target correlations
        plt.figure(figsize=(8, 6))
        feature_target_correlation.plot(kind='bar')
        plt.title("Feature to Target Correlation")
        plt.xlabel("Features")
        plt.ylabel("Correlation coefficient with Target")
        plt.savefig("G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/feature_target_correlation.png")
        plt.close()
    
    def feature_engineering(self, X, fit=False):
        # Feature interactions for highly positive correlations
        interaction_feature = np.atleast_2d(X[:, 0] * X[:, 2]).T
        
        # Combine negatively correlated features
        combined_feature = np.atleast_2d(X[:, 0] / (X[:, 4] + 0.01)).T  # Adding a small constant to avoid division by zero
        
        # Concatenate new features to the original array
        X = np.concatenate((X, interaction_feature, combined_feature), axis=1)
        
        if fit:
            # If fitting the data, initialize the scaler
            self.scaler = StandardScaler().fit(X)
        
            X = self.scaler.transform(X)
            self.features_to_drop = [2, 4]  # These are indexes of features to drop
            X = np.delete(X, self.features_to_drop, axis=1)
            
        print("feature array dimension", X.shape)
        return X
    
    def analyze_model_feedback(self, X_train, y_train, X_val, y_val):
        # Train the model
        self.algorithm.fit(X_train, y_train)
        
        # Get feature importances
        if hasattr(self.algorithm, 'feature_importances_'):
            importances = self.algorithm.feature_importances_
            print(f"Feature importances: {importances}")
        else:
            print("The algorithm does not provide feature importances.")
        
        # Predict on the validation set
        y_pred = self.algorithm.predict(X_val)
        
        # Calculate residuals
        residuals = y_val - y_pred
        
        # Analyze residuals here - as a simple example, let's calculate the mean absolute error
        mean_residual = np.mean(np.abs(residuals))
        print(f"Mean absolute error of the residuals: {mean_residual}")
        
        # Optionally, plot residuals
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals)
        plt.title('Residuals vs Predicted')
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.savefig("G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/residuals_plot.png")
        plt.close()

        # Return feature importances for further analysis if necessary
        return importances if hasattr(self.algorithm, 'feature_importances_') else None
    

    def feature_engineering2(self, X, fit=False):
        # Create interaction features for highly correlated pairs
        # Note that this is a simplification. You should validate each new feature's effectiveness.
        X_interact_02 = np.atleast_2d(X[:, 0] * X[:, 2]).T
        X_interact_45 = np.atleast_2d(X[:, 4] * X[:, 5]).T

        # Create ratio features for negatively correlated pairs
        X_ratio_04 = np.atleast_2d(X[:, 0] / (X[:, 4] + 0.01)).T  

        # Create polynomial features for the features with the highest positive correlation with the target
        X_poly3 = np.atleast_2d(X[:, 3] ** 2).T
        X_poly4 = np.atleast_2d(X[:, 4] ** 2).T
        
        # Concatenate new interaction and ratio features to the original array
        X = np.concatenate((X, X_interact_02, X_interact_45, X_ratio_04), axis=1)
        X = np.concatenate((X, X_poly3, X_poly4), axis=1)

        # Normalizing features - fit the scaler on the training set only
        if fit:
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)

        return X
    
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
        cl = KMeans(n_clusters=cluster, n_init="auto").fit(coord)
        cll = cl.labels_
        
        return cll
    
    def pca_align(self, X_source, X_target1, n_components=6):
        """
        Perform PCA alignment between source and target data.
        """
        # Compute PCA on source domain
        pca_source = PCA(n_components=n_components)
        X_source_pca = pca_source.fit_transform(X_source)
        # pca = PCA().fit(X_source)
        
        """        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Total Explained Variance')
        plt.savefig("G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/pca.png)"""
        
        # Project target domain onto source PCA components
        X_target_pca1 = X_target1.dot(pca_source.components_.T)

        return X_source_pca, X_target_pca1, pca_source.components_
    
    def train_anomaly_detector(self, X_train):
        # Initialize and fit the Isolation Forest
        self.iforest = IsolationForest(random_state=42, contamination=0.1)
        self.iforest.fit(X_train)

    def filter_anomalies(self, X):
        # Predict if a data point is an outlier
        # Returns 1 for inliers, -1 for outliers
        is_inlier = self.iforest.predict(X)
        return X[is_inlier == 1], is_inlier == 1


    def test_extend(self, x, y, test_num, common_mask):

        # Build the test mask
        test_mask = np.zeros_like(self.common_mask).astype(bool)
        test_mask[x, y] = True

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
            
    def dataset_split(self, test_mask_list=None, modify=False):

        if test_mask_list is None:
            test_mask_list = []
            # Randomly choose the start grid
            mask_sum = self.common_mask.sum()
            test_num = int(mask_sum / 5)
            x_arr, y_arr = self.common_mask.nonzero()
            
            positive_x = x_arr[self.label_arr.astype(bool)].reshape((-1,1))
            positive_y = y_arr[self.label_arr.astype(bool)].reshape((-1,1))
            cll = self.km(positive_x, positive_y, self.test_clusters)
            
            for i in range(self.test_clusters):
                cluster_arr = (cll == i)
                
                cluster_x = positive_x[cluster_arr].squeeze()
                cluster_y = positive_y[cluster_arr].squeeze()
                # Throw out the empty array
                if len(cluster_x.shape) == 0:
                    continue
                start = np.random.randint(0, cluster_arr.sum())
                x, y = cluster_x[start], cluster_y[start]
                test_mask = self.test_extend(x, y, test_num, self.common_mask)
                test_mask_list.append(test_mask) 
                
                
        test_mask_list = add_mask(test_mask_list)
        # Buf the test mask
        tmpt = test_mask_list

        # Split the dataset
        dataset_list = []
        print("len(test_mask_list)", len(test_mask_list))
        for test_mask in test_mask_list:
            train_mask = ~test_mask
            test_mask = test_mask & self.common_mask
            train_mask = train_mask & self.common_mask
            train_deposite_mask = train_mask & self.deposit_mask.astype(bool)
            
            # split the val
            mask_sum = train_mask.sum()
            val_num = int(mask_sum / 5)
            x_arr, y_arr = train_mask.nonzero()
            positive_indices = np.argwhere(train_deposite_mask)
            positive_x = np.array([positive_indices[:, 0]]).reshape(-1,1)
            positive_y = np.array([positive_indices[:, 1]]).reshape(-1,1)
            cll = self.km(positive_x, positive_y, self.test_clusters)
 
            
            cluster_arr = (cll == 0)
            cluster_x = positive_x[cluster_arr].squeeze()
            cluster_y = positive_y[cluster_arr].squeeze()

            if np.isscalar(cluster_x) or (isinstance(cluster_x, np.ndarray) and cluster_x.ndim == 0):
                # For example, convert it to a 1-dimensional array
                cluster_x = np.array(cluster_x)
                cluster_x = np.array([cluster_x])
                cluster_y = np.array(cluster_y)
                cluster_y = np.array([cluster_y])
            
            
            start = np.random.randint(0, cluster_arr.sum())
            x, y = cluster_x[start], cluster_y[start]

            val_mask = self.test_extend(x, y, val_num, train_mask)
            val_mask = val_mask[self.common_mask]
            X_val_fold, y_val_fold = self.feature_arr[val_mask], self.total_label_arr[1][val_mask]


            # get the position of the train and test set
            test_pos = np.where(test_mask&self.common_mask)
            test_pos = np.array(test_pos)
            train_pos = np.where(train_mask&self.common_mask)
            train_pos = np.array(train_pos)
            
            train_mask = train_mask[self.common_mask]
            test_mask = test_mask[self.common_mask]
            X_train_fold, X_test_fold = self.feature_arr[train_mask], self.feature_arr[test_mask]
            y_train_fold, y_test_fold = self.total_label_arr[1][train_mask], self.label_arr[test_mask]


     
            test_pos = np.transpose(test_pos)
            X_test_fold = np.concatenate([X_test_fold, test_pos], axis=1)
            train_pos = np.transpose(train_pos)
            X_train_fold = np.concatenate([X_train_fold, train_pos], axis=1)



            # modify testing set
            if modify:
                true_num = y_train_fold.sum()
                index = np.arange(len(y_train_fold))
                true_train = index[y_train_fold == 1]
                false_train = np.random.permutation(index[y_train_fold == 0])[:true_num]
                train = np.concatenate([true_train, false_train])
                X_train_fold = X_train_fold[train]
                y_train_fold = y_train_fold[train]
            
            X_train_fold, y_train_fold = sk_shuffle(X_train_fold, y_train_fold)

            dataset = (X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_fold, y_test_fold)
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
    
    def identify_important_features_with_causal_forest(self, X, y, treatment, N):
        causal_forest = CausalForestDML(model_y=RandomForestRegressor(), model_t=RandomForestClassifier())
        causal_forest.fit(X, y, treatment)
        feature_importances = causal_forest.feature_importances_
        # Select the top N important features
        important_features_indices = np.argsort(feature_importances)[::-1][:N]
        return important_features_indices
            
    def train(self, params,  metrics=['auc', 'pre', 'recall', 'f1'], test_mask=None, modify=False):
        
        modify = self.modify
        if not isinstance(metrics, list):
            metrics = [metrics]
        metric_list = []
        alpha = 0.1
        desired_accuracy = 0.99999  # 99.999% accuracy

        feature_list = ['anticline_distance.tif', 'intersection_distance.tif', 'GneissandSchist_distance.tif', 'GneissandSchist_GodenvilleFormation_distance.tif', 'GneissandSchist_HalifaxFormation_distance.tif', 'GneissandSchist_IgneousRocks_distance.tif', 'GodenvilleFormation_distance.tif', 'GodenvilleFormation_HalifaxFormation_distance.tif', 'HalifaxFormation_distance.tif', 'HalifaxFormation_IgneousRocks_distance.tif', 'IgneousRocks_distance.tif', 'as_ok.tif', 'cu_ok.tif', 'pb_ok.tif', 'zn_ok.tif']

        for metric in metrics:
            if isinstance(metric, str):
                if metric.lower() == 'roc_auc_score' or metric.lower() == 'auc' or metric.lower() == 'auroc':
                    metric = roc_auc_score
                elif metric.lower() == 'f1_score' or metric.lower() == 'f1':
                    metric = f1_score
                elif metric.lower() == 'precision_score' or metric.lower() == 'pre':
                    metric = precision_score
                elif metric.lower() == 'recall_score' or metric.lower() == 'recall':
                    metric = recall_score
     
                else:
                    warnings.warn(f'Wrong metric! Replace it with default metric {Model.DEFAULT_METRIC.__name__}.')
                    metric = Model.DEFAULT_METRIC
            elif isfunction(metric):
                metric = metric
            else:
                warnings.warn(f'Wrong metric! Replace it with default metric {Model.DEFAULT_METRIC.__name__}.')
                metric = Model.DEFAULT_METRIC
            metric_list.append(metric)
            
        score_list, cfm_list, pred_list = [], [], []
        X_fold_sum, shap_values_list = [], []

        if self.mode  == 'IID':
            print("Training with IID mode")
            dataset_list = self.random_spilt(modify=modify)
            
            plt.imshow(self.deposit_mask)
            plt.colorbar()  
            plt.savefig('G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/deposit_mask.png', bbox_inches='tight')
            plt.close()
            
            plt.imshow(self.common_mask)
            plt.colorbar()  
            plt.savefig('G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/common_mask.png', bbox_inches='tight')
            plt.close()
            
            i = 0 
            y_arr_record = []
            conformal_values_list = []
            tpr_list = []
            fpr_list = []
            for dataset in dataset_list:
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_fold, y_test_fold = dataset
                print(X_train_fold.shape)
                print(X_val_fold.shape)
                print(X_test_fold.shape)

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
                        # print("ROC AUC score", score)
                        
                print("'auc', 'pre', 'recall', 'f1'", scores)
                
                tn, fp, fn, tp = confusion_matrix(y_test_fold, pred_arr).ravel()

                tpr = tp / (tp + fn)
                tpr_list.append(tpr)
                fpr = fp / (fp + tn)
                fpr_list.append(fpr)
                print(f"True Positive Rate: {tpr}")
                print(f"False Positive Rate: {fpr}")
                
                

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
            fpr = sum(fpr_list) / len(fpr_list)
            tpr = sum(tpr_list) / len(tpr_list)
            
            print("fpr, tpr", fpr, tpr)
            show_result_map(result_values=y_arr_record, mask=self.common_mask, deposit_mask=self.deposit_mask, test_mask=self.common_mask, index=0, name='concatenate_all_pred',filter=False)

            
        else: 
            
            print("Training with OOD mode")
            # Calculate correlation matrix
            corr_matrix = self.calculate_correlation_matrix()
            # print("corr matrix", corr_matrix)
            
            # Visualize the correlation matrix
            self.visualize_correlation_matrix(corr_matrix)
            # self.feature_engineering()

            feature_target_corr = self.calculate_feature_target_correlation()
            # print("feature_target_corr matrix", feature_target_corr)
            self.visualize_feature_target_correlation(feature_target_corr)

            # Perform feature engineering based on the correlations
            self.feature_arr = self.feature_engineering2(self.feature_arr, fit=True)
            
            test_mask_list, dataset_list = self.dataset_split(modify=modify)
            
            plt.imshow(self.deposit_mask)
            plt.colorbar()  
            plt.savefig('G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/deposit_mask.png', bbox_inches='tight')
            plt.close()
            
            plt.imshow(self.common_mask)
            plt.colorbar()  
            plt.savefig('G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/common_mask.png', bbox_inches='tight')
            plt.close()
            y_arr_record = []
            tpr_list = []
            fpr_list = []
            tpr_after_conformal3 = []
            fpr_after_conformal3 = []
            score_list2 = []
            conformal_values_list_2 = []
            conformal_predictions_record = []
            i = 0
            desired_accuracy = 0.99 
            feature_list = np.array(feature_list)
            importance_list = []
            print(len(dataset_list))
            
            for dataset in dataset_list:
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_fold, y_test_fold = dataset
                print("fold ke ", i+1)
                print("X_train_fold", X_train_fold.shape)
                print("X_val_fold", X_val_fold.shape)
                print("X_test_fold", X_test_fold.shape)
                print("self.feature_arr", self.feature_arr.shape)

                show_dataset_map(X_train_fold,[self.common_mask],i+1,'train')
                show_dataset_map(X_test_fold,[self.common_mask],i+1,'test')
            

                algo = self.algorithm(params)                
                algo.fit(X_train_fold[:, :-2], y_train_fold)
                importances = algo.feature_importances_
                importance_list.append(importances)
                print(f"Feature importances: {importances}")
                
                # Predict on the validation set
                val_pred_arr, val_y_arr = algo.predictor(X_val_fold)
                
                # Calculate residuals
                residuals = y_val_fold - val_y_arr
                
                # Analyze residuals here - as a simple example, let's calculate the mean absolute error
                mean_residual = np.mean(np.abs(residuals))
                print(f"Mean absolute error of the residuals: {mean_residual}")
                
                self.train_anomaly_detector(X_train_fold[:,:-2])

                # Filter out anomalies from the validation and test sets
                X_val_fold, is_inlier_val = self.filter_anomalies(X_val_fold)
                y_val_fold = y_val_fold[is_inlier_val]
                X_test_fold, is_inlier_test = self.filter_anomalies(X_test_fold[:,:-2])
                y_test_fold = y_test_fold[is_inlier_test]
                

                print("X_val_fold", X_val_fold.shape)
                print("X_test_fold", X_test_fold.shape)
                
                # Optionally, plot residuals
                plt.figure(figsize=(10, 6))
                plt.scatter(val_y_arr, residuals)
                plt.title('Residuals vs Predicted')
                plt.xlabel('Predicted values')
                plt.ylabel('Residuals')
                plt.axhline(y=0, color='r', linestyle='-')
                plt.savefig("G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/residuals_plot" + str(i) + ".png")
                plt.close()
       
                scores = []
                
                pred_arr, y_arr = algo.predictor(X_test_fold)
                # set the optimization target
                scores.append(criterion_loss(self.mode, algo, X_val_fold, y_val_fold))
                for metric in metric_list:
                    if metric != roc_auc_score:
                        score = metric(y_true=y_test_fold, y_pred=pred_arr)
                        scores.append(score)
                    else:
                        if len(np.unique(y_test_fold)) > 1:
                            score = metric(y_test_fold, y_arr)
                            scores.append(score)  
                            print("ROC AUC score", score)
                        else:
                            print("ROC AUC score cannot be calculated. Only one class present in y_test_fold.")
                            
                tn, fp, fn, tp = confusion_matrix(y_test_fold, pred_arr).ravel()
                tpr = tp / (tp + fn)
                tpr_list.append(tpr)
                fpr = fp / (fp + tn)
                fpr_list.append(fpr)
                print(f"True Positive Rate: {tpr}")
                print(f"False Positive Rate: {fpr}")
                       
                
                print("'auc', 'pre', 'recall', 'f1'", scores)

                if len(scores) == 1:
                    scores = scores[0]
                    
                score_list.append(scores)
                

                _, y_arr_feature_map = algo.predictor(self.feature_arr)
                y_arr_record.append(y_arr_feature_map)

                _train_mask, _test_mask= get_mask(X_train_fold, X_test_fold, test_mask_list)

                show_result_map(result_values=y_arr_record[i], mask=self.common_mask, deposit_mask=self.deposit_mask, test_mask=self.common_mask, index=i+1, name='all_pred')
                show_result_map(result_values=y_arr_record[i], mask=self.common_mask, deposit_mask=self.deposit_mask, test_mask=_train_mask, index=i+1, name='train_pred')
                show_result_map(result_values=y_arr_record[i], mask=self.common_mask, deposit_mask=self.deposit_mask, test_mask=_test_mask, index=i+1, name='test_pred')
                          
                y_mask = test_mask_list[i].reshape(-1)
                y_mask = y_mask[self.common_mask.reshape(-1)]
                y_arr_record[i][~y_mask] = 0
                
                desired_accuracy_pos = 0.99
                closest_alpha_pos = None
                close_val1_pos = None
                close_val2_pos = None
                closest_accuracy_pos = 0
                closest_accuracy_diff_pos = float('inf')

                desired_accuracy_neg_min = 0.95
                desired_accuracy_neg_max = 0.96
                closest_alpha_neg = None
                close_val1_neg = None
                close_val2_neg = None
                closest_accuracy_neg = 0
                closest_accuracy_diff_neg = float('inf')

                for alpha in np.linspace(0.005, 0.15, 50):
                    # print("alpha", alpha)
                    val1, val2, probabilities= get_qhat2(algo=algo, X_val_fold=X_val_fold, y_val_fold=y_val_fold, alpha2=alpha)
                    
                    # print("val1", val1)
                    # print("val2", val2)
                    conformal_set = get_conformal_set(probabilities=probabilities, feature=X_val_fold, val1=val1, val2=val2)
                    conformal_values = get_conformal_values(conf_sets=conformal_set)
                    
                    # Calculate accuracy
                    pos_acc, neg_acc = new_calculate_accuracy(conformal_values, y_val_fold)
                    # print("pos_acc", pos_acc)
                    # print("neg_acc", neg_acc)
                    
                    # Check and update for positive accuracy
                    accuracy_diff_pos = abs(desired_accuracy_pos - pos_acc)
                    # print("accuracy_diff_pos", accuracy_diff_pos)
                    if accuracy_diff_pos < closest_accuracy_diff_pos :
                        closest_accuracy_diff_pos = accuracy_diff_pos
                        closest_alpha_pos = alpha
                        close_val1_pos = val1
                        closest_accuracy_pos = pos_acc
                    
                        # print("pos", closest_alpha_pos, close_val1_pos, closest_accuracy_pos )
                    
                    # Check and update for negative accuracy
                    accuracy_diff_neg = abs(desired_accuracy_neg_min - neg_acc)
                    # print("accuracy_diff_neg", accuracy_diff_neg)
                    # if desired_accuracy_neg_min <= neg_acc <= desired_accuracy_neg_max and accuracy_diff_neg < closest_accuracy_diff_neg:
                    if accuracy_diff_neg < closest_accuracy_diff_neg:    
                        closest_accuracy_diff_neg = accuracy_diff_neg
                        closest_alpha_neg = alpha
                        close_val2_neg = val2
                        closest_accuracy_neg = neg_acc
                        if accuracy_diff_pos < closest_accuracy_diff_pos and pos_acc >= desired_accuracy_pos:
                            closest_accuracy_diff_pos = accuracy_diff_pos
                            closest_alpha_pos = alpha
                            close_val1_pos = val1
                            closest_accuracy_pos = pos_acc
                            print("val pos renew", close_val1_pos )
                        
                        # print("neg", closest_alpha_neg, close_val2_neg, closest_accuracy_neg, closest_accuracy_pos )
                        
                print("last pos", closest_alpha_pos, close_val1_pos, closest_accuracy_pos ) 
                print("last neg", closest_alpha_neg, close_val2_neg, closest_accuracy_neg ) 
                
                
                truecount_0 = np.count_nonzero(y_test_fold == 0)
                truecount_1 = np.count_nonzero(y_test_fold == 1)
                print("truecount_0", truecount_0)
                print("truecount_1", truecount_1)
                
                avg_thres = (close_val1_pos + close_val2_neg) / 2
                
                zero_fraction = truecount_0/len(y_test_fold)
                print("zero_fraction", zero_fraction)
                
                one_fraction = truecount_1/len(y_test_fold)
                print("one_fraction", one_fraction)
                     
                pos_val = one_fraction * avg_thres 
                neg_val = zero_fraction * avg_thres
                print("pos_val", pos_val)
                print("neg_val", neg_val)

                test_conformal_set3 = get_conformal_set_withprob(algo=algo, feature=X_test_fold, val1=pos_val, val2=neg_val)
                test_conformal_values3 = get_conformal_values(conf_sets=test_conformal_set3)
                coverage_pos_2, coverage_neg_2 = get_coverage(y_val=y_test_fold, conf_sets=test_conformal_set3)
                

                # Calculate accuracy
                pos_acc, neg_acc = new_calculate_accuracy(test_conformal_values3, y_test_fold)
                print("pos_acc_test", pos_acc)
                print("neg_acc_test", neg_acc)
                
                count_05 = np.count_nonzero(test_conformal_values3 == 0.5)
                count_0 = np.count_nonzero(test_conformal_values3 == 0.0)
                count_1 = np.count_nonzero(test_conformal_values3 == 1.0)
                
                # print(len(test_conformal_values3))
                print("0.5", count_05)
                print("0", count_0)
                print("1", count_1)
                # print(count_0+count_05+count_1)
                
  
                # Exclude uncertain predictions
                y_true_filtered3 = [y_test_fold[i] for i in range(len(y_test_fold)) if test_conformal_values3[i] != 0.5]
                y_pred_filtered3 = [pred for pred in test_conformal_values3 if pred != 0.5]
                print("y_true_filtered3", len(y_true_filtered3))
                
                tn, fp, fn, tp = confusion_matrix(y_true_filtered3, y_pred_filtered3).ravel()
                tpr_after = tp / (tp + fn)
                fpr_after = fp / (fp + tn)
                tpr_after_conformal3.append(tpr_after)
                fpr_after_conformal3.append(fpr_after)
            
                # print(y_true_filtered3)
                truecount_0 = y_true_filtered3.count(0)
                truecount_1 = y_true_filtered3.count(1)
                print(truecount_0)
                print(truecount_1)
                
              
                conf_scores = []
                for metric in metric_list:
                    if metric != roc_auc_score:
                        score = metric(y_true=y_true_filtered3, y_pred=y_pred_filtered3)
                        conf_scores.append(score)
                    else:
                        if len(np.unique(y_true_filtered3)) > 1:
                            score = metric(y_true_filtered3, y_pred_filtered3)
                            conf_scores.append(score)  
                            print("ROC AUC score", score)
                        else:
                            print("ROC AUC score cannot be calculated. Only one class present in y_test_fold.")

                
                print(str(i+1)+ "fold" + "confromalized_algo: 'auc', 'pre', 'recall', 'f1', acc", conf_scores)
                
                if len(conf_scores) == 1:
                    conf_scores = conf_scores[0]
                    
                score_list2.append(conf_scores)
                
                conformal_predictions_record.append((y_true_filtered3, y_pred_filtered3))
                
                y_mask = test_mask_list[i].reshape(-1)
                y_mask = y_mask[self.common_mask.reshape(-1)]
                
                all_feature = self.feature_arr
                conformal_set_2 = get_conformal_set_withprob(algo=algo, feature=self.feature_arr, val1 = pos_val, val2 = neg_val)
                # conformal_set_2 = get_conformal_set(algo=algo, feature=X_datas_arrays, val = val1_1)
                coverage_pos_2, coverage_neg_2 = get_coverage(y_val=y_val_fold, conf_sets=conformal_set_2)
                # print("coverage_pos_2", coverage_pos_2)
                # print("coverage_neg_2", coverage_neg_2)

                conformal_values_2 = get_conformal_values(conf_sets=conformal_set_2)
                pos_acc, neg_acc = new_calculate_accuracy(test_conformal_values3, y_test_fold)
                print("pos_acc", pos_acc)
                print("neg_acc", neg_acc)
                
                
                conformal_values_list_2.append(conformal_values_2)
                
                   
                image_conformal(result_values=conformal_values_list_2[i], mask=self.common_mask, deposit_mask=self.deposit_mask, test_mask=_test_mask, index=i+1, name='conformalizedcausal_test_pred')
                image_conformal_withdeposit_forfold(result_values=conformal_values_list_2[i], mask=self.common_mask, deposit_mask=self.deposit_mask, test_mask=_test_mask, index=i+1, name='conformalizedcausal_test_pred_withdeposit')
            
                conformal_values_list_2[i][~y_mask] = 0
                
                i += 1

            y_arr_record = np.maximum.reduce(y_arr_record)
            show_result_map(result_values=y_arr_record, mask=self.common_mask, deposit_mask=self.deposit_mask, test_mask=self.common_mask, index=0, name='concatenate_all_pred',filter=True)
            fpr = sum(fpr_list) / len(fpr_list)
            tpr = sum(tpr_list) / len(tpr_list)
            
            conformal_record_2 = np.maximum.reduce(conformal_values_list_2)
            image_conformal(result_values=conformal_record_2, mask=self.common_mask, deposit_mask=self.deposit_mask, test_mask=self.common_mask, index=0, name='dynamic_all_conformalizedcausal',filter=True)
            image_conformal_withdeposit(result_values=conformal_record_2, mask=self.common_mask, deposit_mask=self.deposit_mask, test_mask=self.common_mask, index=0, name='dynamic_all_conformalizedcausal_withdeposit',filter=True)
            
            
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
    # algo = extAlgo
    algo = rfBoostAlgo
    # algo = svmAlgo
    if algo == rfBoostAlgo:
        x = {'n_estimators': 69, 'learning_rate':0.035}
    elif algo == svmAlgo:
        x = {'C':0.9, 'kernel':'rbf', 'probability': True}
    elif algo == logiAlgo:
        x = {'penalty':'l2', 'max_iter':1000, 'solver':'saga'}
    elif algo == extAlgo:
        x = {'n_estimators': 30, 'max_depth':15}
    elif algo == rfcAlgo:
        x = {'n_estimators': 10, 'max_depth':5}
    elif algo == NNAlgo:
        x = {'max_iter':200, 'hidden_layer_sizes':(15, 30, 10), 'solver':'adam'}
    else:
        x = {'n_estimators': 50}
    print(x)

    task = Model(
        data_path='G:/A PYTHON NOTEBOOK/Bayesian_main/ooddata/Washington.pkl',
        algorithm=algo,
        mode='OOD',
        modify=True,
        test_clusters=5
        )
    
    y = task.evaluate(x)
    print(f'{algo.__name__}: {y}')