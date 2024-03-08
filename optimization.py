import warnings
import time
from multiprocessing import Process, Queue

import numpy as np
from algo import *
from constraints import ParamSpace
from model import Model
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


class Bayesian_optimization:
    """Bayesian optimization class, accepting continuous, discrete, categorical and static parameters of any algorithm for any task. Some metric is required to serve as the criteria of evaluating the performance of hyperparameter configurations. In optimization, a Gaussian Process is utilized to surrogate the original objective function of the target task. The number of objective function calls during optimization is expected to decrease by first acquiring new evaluation configurations through surrogate function.

    Raises:
        RuntimeError: The value of acq_num is out of range.
        RuntimeError: The value of acq_num is not an integer.
        RuntimeError: The value of worker is out of range.
        RuntimeError: The value of worker is not an integer.

    """
    DEFAULT_ACQ_NUM = 1000          # The default number of evaluation points in acquisition 
    DEFAULT_WORKER = 3              # The default number of workers
    WARNING_ACQ_NUM_LOW = 30        # If acq_num is set to be lower than this, a warning will show up
    WARNING_WORKER_HIGH = 20        # If worker is set to be higher than this, a warning will show up
    PARALLEL_LIMIT = False          

    def __init__(self, data_path, task=Model, algorithm=rfcAlgo, mode='random', metrics = ['auc','f1','pre'], default_params=False, cont_book={}, disc_book={}, enum_book={}, stat_book={}, cons_value=10.0, rbf_value=10.0, acq_num=DEFAULT_ACQ_NUM, worker=DEFAULT_WORKER, fidelity=3, modify=False):
        """Initialize the target task, the algorithm to accomplish the task and settings of corresponding hyperparamters

        Args:
            data_path (str): The path of input data files for the target task
            task (class, optional): The target task. Defaults to Model.
            algorithm (class, optional): The algorithm to accomplish the task, with requirement on encapsulation specified in algo.py. Defaults to rfcAlgo.
            default_params (bool, optional): Use the default param settings specified in the algorithm class definition if True. Defaults to False.
            cont_book (dict, optional): The dict of continuous params. Defaults to {}.
            disc_book (dict, optional): The dict of discrete params. Defaults to {}.
            enum_book (dict, optional): The dict of categorical params. Defaults to {}.
            stat_book (dict, optional): The dict of static params. Defaults to {}.
            cons_value (float, optional): Param for ConstantKernel of Gaussian Process. Defaults to 10.0.
            rbf_value (float, optional): Param for RBF of Gaussian Process. Defaults to 10.0.
            acq_num (_type_, optional): The number of evaluation points in acquisition . Defaults to DEFAULT_ACQ_NUM.
            worker (_type_, optional): The number of workers for parallelized Bayesian. Defaults to DEFAULT_WORKER.
            metric(list): The list to record evaluation indicators beside the main one. Defaults to [].
            fidelity (int, optional): The number repeated trials for evaluation. Defaults to 3.
            modify(bool, optional): Whether to downsample the negatives samples to the equal number of positive samples. Defaults to be False.
            path(string): The string to record the name of data estimated.
        """
        self.gaussian = GaussianProcessRegressor(kernel=ConstantKernel(cons_value, constant_value_bounds="fixed") * RBF(rbf_value, length_scale_bounds="fixed"))
        self.task = task(data_path=data_path, algorithm=algorithm, fidelity=fidelity, mode=mode, metrics=metrics, modify=modify)
        
        if default_params:
            self.param_space = ParamSpace(algorithm.DEFAULT_CONTINUOUS_BOOK, algorithm.DEFAULT_DISCRETE_BOOK, algorithm.DEFAULT_ENUM_BOOK, algorithm.DEFAULT_STATIC_BOOK)
        else:
            self.param_space = ParamSpace(cont_book, disc_book, enum_book, stat_book)
        
        self.acq_num = Bayesian_optimization.DEFAULT_ACQ_NUM
        self.set_acq_num(acq_num)
        
        self.worker = Bayesian_optimization.DEFAULT_WORKER
        self.set_worker(worker)

        self.accompany_metric = []
        self.path = data_path
        self.metrics = metrics
        
    def set_acq_num(self, acq_num=DEFAULT_ACQ_NUM):
        """Set the value of acq_num

        Args:
            acq_num (int, optional): The value of acq_num to be set. Defaults to DEFAULT_ACQ_NUM.

        Raises:
            RuntimeError: The value of acq_num is out of range.
            RuntimeError: The value of acq_num is not an integer.
        """
        if not isinstance(acq_num, int):
            raise RuntimeError("The acq_num must be an integer!")
        if acq_num < 1:
            raise RuntimeError(f"The acq_num must be positive, but now it is {acq_num}!")
        if acq_num < Bayesian_optimization.WARNING_ACQ_NUM_LOW:
            warnings.warn(f"The acq_num is suspiciously low. It is {acq_num}.")
            
        self.acq_num = acq_num
        
    def set_worker(self, worker=DEFAULT_WORKER):
        """Set the value of worker

        Args:
            worker (int, optional): The value of worker to be set. Defaults to DEFAULT_WORKER.

        Raises:
            RuntimeError: _The value of worker is out of range.
            RuntimeError: The value of worker is not an integer.
        """
        if not isinstance(worker, int):
            raise RuntimeError("The worker must be an integer!")
        if worker < 1:
            raise RuntimeError(f"The worker must be positive, but now it is {worker}!")
        if worker > Bayesian_optimization.WARNING_WORKER_HIGH:
            warnings.warn(f"The worker is suspiciously high. It is {worker}.")
            
        self.worker = worker
        
    def objective(self, x):
        """Encapsulation of objective function

        Args:
            x (iterable): The evaluation configuration

        Returns:
            float: The score of the input configuration
        """
        x = self.param_space.to_dict(x)
        return self.task.evaluate(x)

    def objective_parallel(self, queue, index, x):
        """Encapsulation for parallelizing the objective function

        Args:
            queue (Queue): Store the output of objective function
            index (int): Label the order of different processes
            x (iterable): The evaluation configuration
        """
        score = self.objective(x)
        queue.put((index, score))
        return

    def surrogate(self, X):
        """Encapsulation of surrogate Gaussian Process

        Args:
            X (Array): The evaluation configurations

        Returns:
            Array: Scores for every configuration in the input
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return self.gaussian.predict(X, return_std=True)
        
    def acquisition(self, X, x_samples):
        """Evaluate the importance of every configuration in the param space

        Args:
            X (Array): Configurations already evaluated by objective function
            x_samples (Array): Configurations to be evaluated by surrogate function

        Returns:
            Array: Surrogate scores of each configuration in x_samples
        """
        y, _ = self.surrogate(X)
        y_best = np.max(y)
        mu, std = self.surrogate(x_samples)
        return norm.cdf((mu - y_best) / (np.array(std) + 1e-9))
    
    def opt_acquisition(self, X):
        """Acquire new evaluation configurations for objective function

        Args:
            X (Array): Configurations already evaluated by objective function

        Returns:
            Array: New evaluation configurations
        """
        x_samples, names = self.param_space.sample(self.acq_num)

        scores = self.acquisition(X, x_samples)
        if self.worker > 1:
            best = (-scores).argsort()[:self.worker]
            return [x_samples[i] for i in best], [names[i] for i in best]
        else:
            best = np.argmax(scores)
            return x_samples[best], names[best]
        
    
    def evaluate_parallel(self, names, x_num):
        """Encapsulation for parallelizing Bayesian

        Args:
            names (iterable): Configurations already evaluated by objective function in name format
            x_num (int): The number of new evaluation configurations required

        Returns:
            Iterable: Scores for each evaluation in names
        """
        queue = Queue()
        p_list = []
        for i in range(x_num):
            p = Process(target=self.objective_parallel, args=[queue, i, names[i]])
            p.start()
            p_list.append(p)
        
        for i in range(x_num):
            p_list[i].join()
            
        index_list = np.zeros((x_num,))
        y = np.zeros((x_num,))
        
        for i in range(x_num):
            index, score = queue.get()
            index_list[i] = index
            accompany = []
            if isinstance(score, float):
                y[i] = score
            else:
                y[i] = score[0]
                accompany.append(score[1:])
                
        if len(accompany) > 0:
            self.accompany_metric = list(np.mean(np.array(accompany), axis=0))
        index = np.argsort(index_list)
        y = y[index]
        
        return y
    
    def evaluate_serial(self, names):
        """Without multi-processing and only one thread
        """
        y, accompany = [], []
        for name in names:
            score = self.objective(name)
            y.append(score[0])
            accompany.append(score[1:])
            if len(accompany) > 0:
                self.accompany_metric = list(np.mean(np.array(accompany), axis=0))
        return np.array(y)

    def initialize(self, x_num=5):
        """Initialize x_num configurations

        Args:
            x_num (int, optional): The number of configurations. Defaults to 5.

        Returns:
            Iterable: Initialized configurations
            Iterable: Scores of initialized configurations
            Iterable: Initialized configurations in name format
        """
        X, names = self.param_space.sample(x_num)   
        y = self.evaluate_parallel(names, x_num) 
        if isinstance(y, list):
            y = y[0]
        self.gaussian.fit(X, y)
    
        return X, y, names
    
    def optimize(self, steps=10, x_num=5, out_log=True, early_stop=0, return_trace=False):
        """Optimize the hyperparamters of the algorithm for the target task

        Args:
            steps (int, optional): The maximal step number. Defaults to 10.
            x_num (int, optional): The number of initialized configurations. Defaults to 5.
            out_log (bool, optional): Print optimization logs if True. Defaults to True.
            early_stop (int, optional): Early stop if no progress in successive specific steps. Disable when set to 0. Defaults to 0.
            return_trace (bool, optional): Return evaluated configurations and scores along with the best configuration if True. Defaults to False.

        Returns:
            Iterable: The best configuration
        """
        start_time = time.time()
        X, y, names = self.initialize(x_num)
        if early_stop == 0:
            early_stop = steps
        early_stop_cnt = 0
        
        # Find the best in initialized samples
        best = np.argmax(y)
        y_best = y[best]
        name_best = names[best]
        
        # Print the best sample in initialized ones
        if out_log:
            now = time.time()
            if len(self.accompany_metric) >= 0:    
                self.param_space.log_head(self.path.split('/')[-1] ,name_best, [y_best] + self.accompany_metric + [now - start_time], self.metrics)
            else:
                self.param_space.log_head(self.path.split('/')[-1] ,name_best, y_best, self.metrics)
                            
        for step_i in range(steps):
            x_sample, name = self.opt_acquisition(X)
            if self.worker > 1:
                if self.PARALLEL_LIMIT:
                    y_ground = self.evaluate_serial(name)
                    worker_best = np.max(np.where(y_ground == np.max(y_ground)))
                else:
                    y_ground = self.evaluate_parallel(name, self.worker)
                    worker_best = y_ground.argmax()
                
                name = name[worker_best]
                x_sample = x_sample[worker_best]
                y_ground = y_ground[worker_best]

            else:
                y_ground = self.objective(name)
            # y_sample, _ = self.surrogate(X)

            X = np.concatenate((X, np.array([x_sample])))
            y = np.concatenate((y, np.array([y_ground]).reshape(1,)))

            # Check whether this is the best score till now
            flag = False
            if y_ground > y_best:
                y_best = y_ground
                name_best = name
                early_stop_cnt = 0
                flag = True
            
            # Print the result of the current epoch
            if out_log:
                if step_i > 0:
                    now = time.time()
                    if len(self.accompany_metric) > 0:
                        y_ground = [y_ground] + self.accompany_metric
                    self.param_space.log_out(self.path.split('/')[-1], name, y_ground + [now - start_time], flag)
            
            # Early stop
            if not flag:
                early_stop_cnt += 1
                if early_stop_cnt == early_stop:
                    break
            else:
                early_stop_cnt = 0
                
            # Update the surrogate function
            self.gaussian.fit(X, y)
            
        if return_trace:
            return name_best, X, y
        else:
            return name_best