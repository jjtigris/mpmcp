import pickle
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from algo import *
from model import Model


def get_score(score_list):
    y = np.mean(np.array(score_list)[:,0])
    return y

class Feature_Filter:
    def __init__(self, input_feature_arr=None, data_path=None, params={'n_estimators': 10, 'max_depth': 10}, algorithm=rfcAlgo, epoches=1, out_log=True):
        """    
        Args:
            dataset (tuple, optional): Features, labels, mask and feature name list. Prior to data_path. Defaults to None.
            data_path (str, optional): Path of input data file. Only used when dataset is None. Defaults to None.
            params (dict, optional): The param configuration used. Defaults to {}.
            algorithm (class, optional): The algorithm for the target task. Defaults to rfcAlgo.
            epoches (int, optional): The number of repeated trials for evaluation. Defaults to 5.
            out_log (bool, optional): Print the log if True. Defaults to True.

        Raises:
            RuntimeError: Both dataset and data_path are None.
        """
        self.data_path = data_path
        self.params = params
        self.algorithm = algorithm
        self.epoches = epoches
        self.out_log = out_log

        if input_feature_arr is not None:
            self.feature_arr = input_feature_arr
        elif self.data_path is not None:
            with open(self.data_path, 'rb') as f:
                self.feature_arr, _, _, _ = pickle.load(f)
        else:
            raise RuntimeError("No data is specified in get_shapley!")

    def get_shapley(self, dataset=None):
        """Get shapley value of each feature in the dataset

        Args:
            dataset (tuple, optional): Features, labels, mask and feature name list. Prior to data_path. Defaults to None.

        Raises:
            RuntimeError: Both dataset and data_path are None.

        Returns:
            list: Shapley value of each feature in the dataset.
        """
        feature_arr = self.feature_arr

        def calculate_shapley(i):
            model = Model(data_path=self.data_path, algorithm=self.algorithm, modify=True)
            model.feature_arr = np.array(feature_arr[:, i]).reshape(-1, 1)
            print(f"Calculating Shapley for feature {i}...")
            score_list = [model.train(params=self.params)[0] for _ in range(self.epoches)]
            return i, np.mean(score_list)

        feature_len = feature_arr.shape[1]
        score_dict = {}

        # Training the baseline model using all features
        model = Model(data_path=self.data_path, algorithm=self.algorithm, modify=True)
        score_list = [np.mean(model.train(params=self.params)[0]) for _ in range(self.epoches)]
        score_dict[tuple(range(feature_len))] = np.mean(score_list)
        del model

        model = Model(data_path=self.data_path, algorithm=self.algorithm, modify=True)
        model.feature_arr = np.zeros(shape=feature_arr.shape)
        score_list = [np.mean(model.train(params=self.params)[0]) for _ in range(self.epoches)]
        score_dict[tuple()] = np.mean(score_list)
        del model

        # Calculate Shapley values for each feature
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(calculate_shapley, i) for i in range(feature_len)]
            for future in futures:
                i, score = future.result()
                key = (i,)  
                if key not in score_dict:
                    score_dict[key] = 0
                score_dict[key] += score / self.epoches

        shapley_list = [0] * feature_len
        for i in range(feature_len):
            key = (i,)
            shapley_list[i] = (score_dict[key] - score_dict[()]) / feature_len

        if self.out_log:
            for i, shapley_value in enumerate(shapley_list):
                print(f"Shapley value of feature {i}: {shapley_value}")

        return shapley_list
    
    def select_top_features(self, top_k=20):
        """Select the top-k features 

        Args:
            shapley_list (list): List of Shapley values for each feature.
            top_k (int, optional): Number of top features to select. Defaults to 20.

        Returns:
            list: List of indices of the top-k features.
        """

        shapley_list = self.get_shapley(self.data_path)
        shapley_tuples = [(i, shapley_value) for i, shapley_value in enumerate(shapley_list)]
        shapley_tuples.sort(key=lambda x: x[1], reverse=True)
        # Extract the indices of the top-k features
        top_feature_indices = list(set([i for i, _ in shapley_tuples[:top_k]] + [i for i in range(self.feature_arr.shape[1])]))
        new_feature_arr = self.feature_arr[:, top_feature_indices]

        if self.out_log:
            print("Selected Feature Indices: \n", top_feature_indices)
        return new_feature_arr