from algo import *
from optimization import Bayesian_optimization
import numpy as np
from model import Model
from multiprocessing import Process, Manager
import concurrent.futures

"""
Here is a new method to choose a better machine learning model before the main preocess to optimize
To balance the performance and time consumption, Each method to be selected will be iterated 3 times, 
and the best score obtained will be used as the evaluation criterion .The model that gets the highest
score will be selected in the follow up process

This feature can be enabled in test.py by calling Method_select class
"""

class Method_select:
    def __init__(self, algorithms=[rfcAlgo, NNAlgo]):
        self.algos = algorithms
        self.best_algo = None
        self.opt_score = -100

    def evaluate_algo(self, algo, data_path, task, mode):
        # low-fidelity estimation for method selection
        
        bo = Bayesian_optimization(data_path, task, algo, mode=mode, default_params=True, fidelity=1, worker=3, modify=True)
        best, X, y = bo.optimize(steps=5, out_log=False, return_trace=True)
        score = np.mean(y)
        print(f'{algo.__name__}, score: {score:.4f}')
        return score

    def select(self, data_path, task, mode):

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_algo = {executor.submit(self.evaluate_algo, algo, data_path, task, mode): algo for algo in self.algos}
            
            for future in concurrent.futures.as_completed(future_to_algo):
                algo = future_to_algo[future]
                try:
                    score = future.result()
                    if score > self.opt_score:
                        self.best_algo = algo
                        self.opt_score = score
                        
                except Exception as exc:
                    print(f'Error while evaluating {algo.__name__} Model: {exc}')

        return self.best_algo
    