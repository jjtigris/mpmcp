from utils3 import *

import os
os.chdir('Bayesian_main')
 

if __name__ == '__main__':

    data_dir = 'G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/dataset'
    # output_dir = './data'
    # preprocess_all_data(output_dir='G:/A PYTHON NOTEBOOK/Bayesian_main/ooddata', target_name='Au', label_filter=True)
    # preprocess_all_data(data_dir=data_dir, output_dir='G:/A PYTHON NOTEBOOK/Bayesian_main/ooddata', target_name='Au', label_filter=True)
    # preprocess_data_interpolate(method='linear')
    preprocess_data_interpolate(data_dir='G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/dataset/Washington', method='linear')
    # preprocess_Nova_data(data_dir="G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/dataset/NovaScotia2", output_path = 'G:/A PYTHON NOTEBOOK/Bayesian_main/ooddata/Nova.pkl')
    # preprocess_Nova_data(data_dir, feature_prefix='', feature_suffix='.npy', mask_dir='Mask.npy', label_path_list=['Target.npy'], augment=False, output_path = './data_benchmark/Nova.pkl')



