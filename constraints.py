import numpy as np
import warnings
from datetime import datetime
"""
Continuous Param requires a floating point list length 2 as the lower and upper bound
Discrete Param requires an integer list legnth 2 as the lower and upper bound
Categorical Param requires a list as the enumeration of all feasible options
Static Param requires a value as the static value
ParamSpace stores all param settings and generates samples.
"""

DEFAULT_CONTINUOUS_PARAM_LOW = 0.0
DEFAULT_CONTINUOUS_PARAM_HIGH = np.inf
DEFAULT_DISCRETE_PARAM_LOW = 0
DEFAULT_DISCRETE_PARAM_HIGH = 1000

class WrongClassError(RuntimeError): ...
    
class Param:
    def __init__(self):
        pass
    
    def sample(self):
        pass    

class ContinuousParam(Param):
    def __init__(self, key='ContinuousParam', value=(DEFAULT_CONTINUOUS_PARAM_LOW, DEFAULT_CONTINUOUS_PARAM_HIGH)):
        super().__init__()
        self.name = key
        
        # Check the format of value
        if not isinstance(value, list):
            raise WrongClassError(f"The bound value of param {key} is not a list!")
        if len(value) > 2:
            raise RuntimeError(f'The length of bound value of param {key} should be no more than 2, which is {len(value)}!')
        if len(value) == 0:
            value.append(DEFAULT_CONTINUOUS_PARAM_LOW)
        if len(value) == 1:
            value.append(DEFAULT_CONTINUOUS_PARAM_HIGH)
        if not isinstance(value[0], float):
            raise WrongClassError(f'The lower bound of param {key} should be floating point, which is {value[0].__class__}!')
        if not isinstance(value[1], float):
            raise WrongClassError(f'The upper bound of param {key} should be floating point, which is {value[1].__class__}!')
        
        self.low = value[0]
        self.high = value[1]
        
    def sample(self):
        return np.random.uniform(low=self.low, high=self.high)
        
class DiscreteParam(Param):
    def __init__(self, key='DiscreteParam', value=(DEFAULT_DISCRETE_PARAM_LOW, DEFAULT_DISCRETE_PARAM_HIGH)):
        super().__init__()
        self.name = key

        # Check the format of value
        if not isinstance(value, list):
            raise WrongClassError(f"The bound value of param {key} is not a list!")
        if len(value) > 2:
            raise RuntimeError(f'The length of bound value of param {key} should be no more than 2, which is {len(value)}!')
        if len(value) == 0:
            value.append(DEFAULT_DISCRETE_PARAM_LOW)
        if len(value) == 1:
            value.append(DEFAULT_DISCRETE_PARAM_HIGH)
        if not isinstance(value[0], int):
            raise WrongClassError(f'The lower bound of param {key} should be floating point, which is {value[0].__class__}!')
        if not isinstance(value[1], int):
            raise WrongClassError(f'The upper bound of param {key} should be floating point, which is {value[1].__class__}!')
            
        self.low = value[0]
        self.high = value[1]
    
    def sample(self):
        return np.random.randint(low=self.low, high=self.high)
    
class EnumParam(Param):
    def __init__(self, key='EnumParam', value=['default']):
        super().__init__()
        self.name = key
        
        # Check the format of value
        if not isinstance(value, list):
            raise WrongClassError(f"The bound value of param {key} is not a list!")
        if len(value) == 0:
            raise RuntimeError(f'The member list of param {key} should not be empty!')
        
        self.num_member = len(value)
        self.member = value    

    def sample(self):
        s = np.random.randint(low=0, high=self.num_member)
        return s, self.member[s]

class StaticParam(Param):
    def __init__(self, key='StaticParam', value=0):
        super().__init__()
        self.name = key

        # Check the format of value
        if value is None:
            raise WrongClassError(f"The static value of param {key} should not be None!")
        
        self.value = value

    def sample(self):
        return self.value
    
class ParamSpace:
    def __init__(self, cont_book, disc_book, enum_book, stat_book):
        self.param_list = []
        self.name_list = []
        self.index = 0
        
        # Check the format of continuous params
        if not isinstance(cont_book, dict):
            raise WrongClassError(f"Cont_book should be dict, which is {cont_book.__class__}")
        for key, value in cont_book.items():
            self.name_list.append(key)
            param = ContinuousParam(key, value)
            self.param_list.append(param)
        
        # Check the format of discrete params
        if not isinstance(disc_book, dict):
            raise WrongClassError(f"Disc_book should be dict, which is {disc_book.__class__}")
        for key, value in disc_book.items():
            self.name_list.append(key)
            param = DiscreteParam(key, value)
            self.param_list.append(param)
            
        # Check the format of enumerate params
        if not isinstance(enum_book, dict):
            raise WrongClassError(f"Enum_book should be dict, which is {enum_book.__class__}")
        for key, value in enum_book.items():
            self.name_list.append(key)
            param = EnumParam(key, value)
            self.param_list.append(param)
            
        # Check the format of static params
        if not isinstance(stat_book, dict):
            raise WrongClassError(f"Stat_book should be dict, which is {stat_book.__class__}")
        for key, value in stat_book.items():
            self.name_list.append(key)
            param = StaticParam(key, value)
            self.param_list.append(param)
        
        if len(self.param_list) == 0:
            warnings.warn('No param is specified!')
            
    def sample(self, x_num=1):
        sample_matrix = []
        name_matrix = []
        for _ in range(x_num):
            sample_row = []
            name_row = []
            for param in self.param_list:
                if isinstance(param, EnumParam):
                    no, name = param.sample()
                    sample_row.append(no)
                    name_row.append(name)
                else:
                    no = param.sample()
                    sample_row.append(no)
                    name_row.append(no)
            sample_matrix.append(sample_row)
            name_matrix.append(name_row)
            
        return np.array(sample_matrix), name_matrix

    def log_head(self, name, x, y, metrics):
        """Print the table head of log

        Args:
            x (Iterable): The first best param, usually from initialization of optimization
            y (float): Score for x
        """

        self.index = 1
        now = datetime.now().strftime('%Y%m%d_%H_%M_%S')
        self.now = now
        print("in the log head")
        with open(f"G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/{name.replace('.pkl','_')}{self.now}_log.md", "w") as f:
            f.write("idx | ")
            for i in range(len(self.name_list)):
                f.write(f"{self.name_list[i]} | ")
            length = len(metrics) + 2
            f.write("Train Score | ")
            for i in range(length):
                try:
                    if metrics[i] == 'auc':
                        f.write("AUC Score | ")
                    if metrics[i] == 'f1':
                        f.write("F1 Score | ")
                    if metrics[i] == 'pre':
                        f.write("Precision | ")
                    if metrics[i] == 'recall':
                        f.write("Recall | ")
                except:
                    metric_name = metrics[i-len(metrics)].upper()
                    f.write(f"{metric_name} std | ")

            f.write("Time Cost | \n")
            f.write("--- | ")
            for i in range(len(self.name_list)):
                f.write(f"--- | ")
            f.write("--- | --- | --- | --- | --- | --- | \n")
        self.log_out(name, x, y, True)

            
    def log_out(self, name, x, y, flag, index=None):
        """Print the log for one optimization step

        Args:
            x (iterable): The best param in current step
            y (float): Score for x
            flag (bool): A new best param if True, which means requiring bold font
            index (int, optional): Pre-set index, overwriting the index in use. Defaults to None.
        """
        print("in log out")
        bold = "**" if flag else ""
        if index is None:
            index = self.index
        else:
            self.index = index
        with open(f"G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/{name.replace('.pkl','_')}{self.now}_log.md", "a") as f:
            f.write(f"{index} |")
            for x_i in x:
                f.write(f"{bold}{x_i}{bold} | ")
                
            if isinstance(y, list):
                for y_i in y:
                    f.write(f"{bold}{y_i:.4}{bold} | ")
            else:
                f.write(f"{bold}{y:.4}{bold} | ")
            f.write(f"\n")
        self.index += 1
    
    def to_dict(self, x):
        """Convert a configuration in iterable format to a dict format

        Args:
            x (iterable): A configuration

        Returns:
            dict: The configuration in dict format
        """
        return {n: v for n,v in zip(self.name_list, x)}