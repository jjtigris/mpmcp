import rasterio
import geopandas
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pylab
import heapq
from queue import Queue as pyQueue
import scipy.stats as ss
from scipy.interpolate import interp2d
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, auc, roc_curve, make_scorer, mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from scipy.ndimage import median_filter
from sklearn.metrics import roc_auc_score
from optimization import Bayesian_optimization
# import ListedColormap
from matplotlib.colors import ListedColormap
from algo import *        
# import pykrige
import time
import os
import sys
# from metric import Feature_Filter
from interpolation import interp_opt

"""
The early stage data preprocess and some plot functions.

"""
from metric import Feature_Filter

def preprocess_data(data_dir='./dataset/nefb_fb_hlc_cir', feature_list=['As', 'B', 'Ca', 'Co', 'Cr', 'Cu', 'Fe', 'Mg', 'Mn', 'Ni', 'Pb', 'Sc', 'Y'], feature_prefix='', feature_suffix='.tif', mask='raster/mask1.tif', target_name='Au', label_path_list=['shape/nefb_fb_hlc_cir_Deposites.shp', 'shape/nefb_fb_hlc_cir_deposit_Au_quarzt.shp', 'shape/nefb_fb_hlc_cir_deposit_Ploymetallic_vein.shp'], augment=True, label_filter=True, feature_filter=False, output_path='./data/nefb_fb_hlc_cir.pkl'):
    """Preprocess the dataset from raster files and shapefiles into feature, label and mask data

    Args:
        data_dir (str, optional): The directory of raw data. Defaults to '../../dataset/nefb_fb_hlc_cir'.
        feature_list (list, optional): The list of features to be used. Defaults to ['As', 'B', 'Ca', 'Co', 'Cr', 'Cu', 'Fe', 'Mg', 'Mn', 'Ni', 'Pb', 'Sc', 'Y'].
        feature_prefix (str, optional): The prefix before the feature name in the path of feature raw data. Defaults to ''.
        feature_suffix (str, optional): The suffix behind the feature name in the path of feature raw data. Defaults to '.tif'.
        mask (str, optional): The path of mask raw data. Defaults to 'raster/mask1.tif'.
        target_name (str, optional): The name of target. Defaults to 'Au'.
        label_path_list (list, optional): The list of path of label raw data. Defaults to ['shape/nefb_fb_hlc_cir_Deposites.shp', 'shape/nefb_fb_hlc_cir_deposit_Au_quarzt.shp', 'shape/nefb_fb_hlc_cir_deposit_Ploymetallic_vein.shp'].
        augment (bool, optional): Whether to perform data augment operations. Defaults to True.
        label_filter (bool, optional): Whether to fileter the label raw data before process. Defaults to True.
        feature_filter (bool, optional): Whether to fileter the raw features before process instead of using feature list. Defaults to False.
        output_path (str, optional): The path of output data files. Defaults to '../data/nefb_fb_hlc_cir.pkl'.

    Returns:
        Array: The array of samples' feature
        Array: The array of samples' label
        Array: The array of mask
        list: The list of features' name
    """
    
    # Load feature raw data
    feature_dict = {}
    for feature in feature_list:
        rst = rasterio.open(data_dir+f'/{feature_prefix}{feature}{feature_suffix}')
        feature_dict[feature] = rst.read(1)
        
    # Load mask raw data and preprocess
    mask_ds = rasterio.open(data_dir+f'/{mask}')
    mask_data = mask_ds.read(1)
    mask = make_mask(data_dir, mask_data)
    
    # More features added and filtered 
    if feature_filter:
        dirs = os.listdir(data_dir + '/TIFs')
        for feature in dirs:
            if 'tif' in feature:
                if 'toline.tif' in feature:
                    continue
                rst = rasterio.open(data_dir + '/TIFs/' + feature).read(1)
                if rst.shape != mask.shape:
                    continue
                feature_list.append(feature)
                feature_dict[feature] = np.array(rst) 

    # Preprocess feature
    feature_arr = np.zeros((mask.sum(),len(feature_list)))
    for i, feature in enumerate(feature_list):
        feature_arr[:, i] = feature_dict[feature][mask]
        
    # Load label raw data
    label_x_list = []
    label_y_list = []
    for path in label_path_list:
        deposite = geopandas.read_file(data_dir+f'/{path}')
        # Whether to filter label raw data
        if label_filter:
            deposite = deposite.dropna(subset='comm_main')
            au_dep = deposite[[target_name in row for row in deposite['comm_main']]]
        else:
            au_dep = deposite
        # Extract the coordinate
        label_x = au_dep.geometry.x.to_numpy()
        label_y = au_dep.geometry.y.to_numpy()

        label_x_list.append(label_x)
        label_y_list.append(label_y)

    # Preprocess label
    x = np.concatenate(label_x_list)
    y = np.concatenate(label_y_list)
    row, col = mask_ds.index(x,y)
    row_np = np.array(row)
    row_np[row_np == mask_data.shape[0]] = 1
    label_arr2d = np.zeros_like(mask_data)
    for x, y in zip(row_np, col):
        label_arr2d[x, y] = 1

    deposite_mask = label_arr2d
    ground_label_arr = label_arr2d[mask]
    label_arr = ground_label_arr
    # Data augment
    if augment:
        label_arr2d = augment_2D(label_arr2d)
        label_arr = label_arr2d[mask]
    
    # feature filtering
    if feature_filter:
        feature_filter_model = Feature_Filter(input_feature_arr=feature_arr)
        feature_arr = feature_filter_model.select_top_features(top_k=20)

    # Pack and save dataset
    dataset = (feature_arr, np.array([ground_label_arr, label_arr]), mask, deposite_mask)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)


def preprocess_all_data(data_dir='./dataset', output_dir='./data', target_name='Au', label_filter=True, augment=False):
    preprocess_data(
        data_dir=f'{data_dir}/nefb_fb_hlc_cir', 
        feature_list=['As', 'B', 'Ca', 'Co', 'Cr', 'Cu', 'Fe', 'Mg', 'Mn', 'Ni', 'Pb', 'Sc', 'Y'], 
        feature_prefix='raster/', 
        feature_suffix='.tif', 
        mask='raster/mask1.tif', 
        target_name=target_name, 
        label_path_list=['shape/nefb_fb_hlc_cir_Deposites.shp', 'shape/nefb_fb_hlc_cir_deposit_Au_quarzt.shp', 'shape/nefb_fb_hlc_cir_deposit_Ploymetallic_vein.shp'], 
        output_path=f'{output_dir}/nefb_fb_hlc_cir.pkl',
        label_filter=label_filter,
        augment=augment
        )
    
    preprocess_data(
        data_dir=f'{data_dir}/tok_lad_scsr_ahc', 
        feature_list=['B', 'Ca1', 'Co1', 'Cr1', 'Cu', 'Fe', 'Mg', 'Mn', 'Ni', 'Pb', 'Sc', 'Sr', 'V', 'Y', 'Zn'], 
        feature_prefix='raster/', 
        feature_suffix='.tif', 
        mask='raster/mask.tif', 
        target_name=target_name, 
        label_path_list=['shape/tok_lad_scsr_ahc_Basaltic_Cu_Au.shp','shape/tok_lad_scsr_ahc_porphyry_Cu_Au.shp', 'tok_lad_scsr_ahc_Placer_Au.shp'], 
        output_path=f'{output_dir}/tok_lad_scsr_ahc.pkl',
        label_filter=label_filter,
        augment=augment
        )
    
    preprocess_data(
        data_dir=f'{data_dir}/North Idaho', 
        feature_list=['ba', 'ca', 'cr', 'cu', 'fe', 'la', 'mg', 'mn', 'ni', 'pb', 'sr', 'ti', 'v', 'y', 'zr'], 
        feature_prefix='Raster/Geochemistry/', 
        feature_suffix='', 
        mask='Raster/Geochemistry/pb', 
        target_name=target_name, 
        label_path_list=['Shapefiles/Au.shp'], #, 'Shapefiles/mineral_deposit.shp'
        output_path=f'{output_dir}/North_Idaho.pkl',
        label_filter=False,
        augment=augment
        )
    
    preprocess_data(
        data_dir=f'{data_dir}/bm_lis_go_sesrp', 
        feature_list=['ag_ppm', 'as', 'be_ppm', 'ca', 'co', 'cr', 'cu', 'fe', 'la', 'mg', 'mn', 'ni', 'pb', 'ti'], 
        feature_prefix='raster/', 
        feature_suffix='', 
        mask='raster/mask.tif', 
        target_name=target_name, 
        label_path_list=['shapefile/bm_lis_go_quartzveinsAu.shp'], 
        output_path=f'{output_dir}/bm_lis_go_sesrp.pkl',
        label_filter=label_filter,
        augment=augment
        )
    
    
def preprocess_data_interpolate(data_dir='./dataset/Washington', augment:bool = True, method = 'kriging', feature_filter = False):
    """
    Convert point data to raster data by interpolation

    Args:
        data_dir (str, optional): The directory of raw data. 
        augment (bool, optional): Whether to perform data augment operations. Defaults to True.
        method (str, optional): The method for interpolation
        feature_filter (bool, optional): Whether to fileter the raw features before process instead of using feature list. Defaults to False.
        
    Returns:
        Array: The array of samples' feature
        Array: The array of samples' label
        Array: The array of mask
        list: The list of features' name
    """
    
    mask_ds = rasterio.open(data_dir+'/shapefile/mask1.tif')
    mask_data = mask_ds.read(1)
    mask = mask_data == 1
    
    au = geopandas.read_file(data_dir+'/shapefile/Au.shp')
    x = au.geometry.x.to_numpy()
    y = au.geometry.y.to_numpy()
    row, col = mask_ds.index(x,y)

    row_np = np.array(row)
    row_np[np.array(row) == mask_data.shape[0]] = 1
    label_arr2d = np.zeros_like(mask_data)
    for x, y in zip(row_np, col):
        label_arr2d[x, y] = 1
    
    deposite_mask = label_arr2d
    ground_label_arr = label_arr2d[mask]
    if augment:
        label_arr2d = augment_2D(label_arr2d)
    label_arr = label_arr2d[mask]
    
    geochemistry = geopandas.read_file(data_dir+'/shapefile/Geochemistry.shp')   
    feature_list = ['B', 'Ca', 'Cu', 'Fe', 'Mg', 'Ni']
    feature_dict = {}
    size = mask_ds.index(mask_ds.bounds.right, mask_ds.bounds.bottom)
    for feature in feature_list:
        feature_data = np.zeros(size)
        for i, row in geochemistry.iterrows():
            x = row.geometry.x
            y = row.geometry.y
            x, y = mask_ds.index(x, y)
            data = row[feature]
            if data < 1e-8:
                data = 1e-8
            feature_data[x, y] = data
            feature_dict[feature] = feature_data
        
    x_geo, y_geo = geochemistry.geometry.x.values, geochemistry.geometry.y.values
    x_max, y_max = mask_ds.index(mask_ds.bounds.right, mask_ds.bounds.bottom)

    # Interpolation to transfer shapfiles to rater form
    for feature in feature_list:
        print(f'Processing {feature}')
        z = geochemistry[feature].values

        # interp optimization
        interpOPT = interp_opt()
        result = interpOPT.optimize(x_geo, y_geo, z, x_max, y_max)

        feature_dict[feature] = result
            
    feature_arr2d_dict = feature_dict.copy()
    feature_arr = np.zeros((mask.sum(),len(feature_list)))
    for idx in range(len(feature_list)):
        feature_arr[:,idx] = feature_arr2d_dict[feature_list[idx]][mask]

    if feature_filter:
        feature_filter_model = Feature_Filter(input_feature_arr=feature_arr)
        feature_arr = feature_filter_model.select_top_features(top_k=20)

    dataset = (feature_arr, np.array([ground_label_arr, label_arr]), mask, deposite_mask)
    with open(f'./data/Washington_{method}.pkl', 'wb') as f:
        pickle.dump(dataset, f)

def preprocess_Nova_data(data_dir, feature_prefix='', feature_suffix='.npy', mask_dir='Mask.npy', label_path_list=['Target.npy'], augment=False, output_path = './data_benchmark/Nova.pkl'):
    # Process the NovaScotia2 Data
    feature_list = ['Godenville_Formation_Buffer', 'Anticline_Buffer', 'As', 'Li', 'Pb', 'F', 'Cu', 'W', 'Zn']
    feature_dict = {}
    for feature in feature_list:
        rst = np.load(data_dir+f'/{feature_prefix}{feature}{feature_suffix}')
        feature_dict[feature] = rst
        
    
    # Load mask data and preprocess
    mask = np.load(data_dir+ '/' +mask_dir).astype(np.int64)
    mask = make_mask(data_dir, mask_data=mask, show=True)

    # Preprocess features
    feature_arr = np.zeros((mask.sum(), len(feature_list)))
    for i, feature in enumerate(feature_list):
        feature_arr[:, i] = feature_dict[feature][mask]
    
    # Load the target ID
    label_arr = np.zeros(shape=(feature_arr.shape[0], ))
    
    for path in label_path_list:
        depositMask = (np.load(data_dir + '/' + path) > 0 )
        ground_label_arr = depositMask[mask]
        label_arr = ground_label_arr

        if augment:
            label_arr2d = augment_2D(depositMask) 
            label_arr = label_arr2d[mask]
    
    dataset = (feature_arr, np.array([ground_label_arr, label_arr]), mask, depositMask)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    return dataset


def make_mask(data_dir, mask_data, show =False):

    if 'nefb' in data_dir or 'tok' in data_dir or 'Washington' in data_dir:
        mask = mask_data != 0
    
    elif 'bm' in data_dir:
        mask = mask_data == 1

    elif 'North' in data_dir:
        mask = (mask_data > -1)
    
    else:
        mask = mask_data != 0

    if show:
        plt.figure()
        plt.imshow(mask)
        plt.colorbar()
        name = data_dir.replace('/','')
        plt.title("Mask")
        # plt.savefig(f'./backup/mask_{name}.png')

    return mask

def augment_2D(array, wide_mode = False):
    """
    For data augment function. Assign the 3*3 blocks around the sites to be labeled.
    """
    new = array.copy()
    a = np.where(array == 1)
    x, y = a[0], a[1]
    aug_touple = [(-1,-1),(-1,1),(1,-1),(1,1),(0,1),(0,-1),(1,0),(-1,0)]
    print(array.sum())
    if wide_mode:
        aug_touple = [
            (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
            (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
            (0, -2),  (0, -1),  (0, 0),  (0, 1),  (0, 2),
            (1, -2),  (1, -1),  (1, 0),  (1, 1),  (1, 2),
            (2, -2),  (2, -1),  (2, 0),  (2, 1),  (2, 2),
        ]

    for idx in range(len(x)):
        for m,n in aug_touple:
            newx = x[idx] + m
            newy = y[idx] + n
            
            if (0< newx and newx < array.shape[0]) and (0< newy and newy < array.shape[1]):
                new[newx][newy] = 1
    return new


def plot_roc(fpr, tpr, scat=False, save=True, mode='normal'):
    """
        plot ROC curve
    """
    roc_auc = auc(fpr, tpr)
    # plt.plot(fpr, tpr, label=f'split set {index}, ROC area = {roc_auc:.2f}', lw=2)
    plt.plot(fpr, tpr, label=f' ROC area = {roc_auc:.2f}', lw=2)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    if scat:
        plt.scatter(fpr, tpr)
    plt.title("ROC curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

    if save:
        print("saving")
        plt.grid(alpha=0.8)
        plt.legend()
        plt.tight_layout()
        # Save the figure
        file_path = os.path.abspath(__file__)
        file_dir = os.path.dirname(file_path)
        path_save = os.path.join(file_dir, '/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/')

        plt.savefig(path_save + mode + 'roc' + '.png', dpi=300)
        # plt.savefig(f'Bayesian_main/run/figs/roc.png')
    else:
        plt.show()    


def plot_PR(y_test_fold, y_arr, index, mode):
    """
    plot Precision-Recall curve
    """
    prec, recall, _ = precision_recall_curve(y_test_fold, y_arr)
    non_zero_indices = np.logical_and(prec != 0, recall != 0)
    f1_scores = 2 * (prec[non_zero_indices] * recall[non_zero_indices]) / (prec[non_zero_indices] + recall[non_zero_indices])
    max_f1_score = np.max(f1_scores)
    max_f1_score_index = np.argmax(f1_scores)
    plt.plot(recall, prec, label = f'split set: {index}, Max F1: {max_f1_score:.2f}')
    plt.scatter(recall[max_f1_score_index], prec[max_f1_score_index], c='red', marker='o')
    plt.legend()
    plt.grid(alpha=0.8)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve for mineral prediction")
    plt.tight_layout()
    
    file_path = os.path.abspath(__file__)
    file_dir = os.path.dirname(file_path)
    path_save = os.path.join(file_dir, '/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/')

    plt.savefig(path_save+ mode + 'precision-recall' +'_'+str(index)+'.png', dpi=300)

    # plt.savefig('Bayesian_main/run/figs/precision-recall.png', dpi=300)
    
def plot_PR2(y_test_fold, y_arr, index, mode):
    """
    plot Precision-Recall curve
    """
    prec, recall, _ = precision_recall_curve(y_test_fold, y_arr)
    non_zero_indices = np.logical_and(prec != 0, recall != 0)
    f1_scores = 2 * (prec[non_zero_indices] * recall[non_zero_indices]) / (prec[non_zero_indices] + recall[non_zero_indices])
    max_f1_score = np.max(f1_scores)
    max_f1_score_index = np.argmax(f1_scores)
    plt.plot(recall, prec, label = f'split set: {index}, Max F1: {max_f1_score:.2f}')
    plt.scatter(recall[max_f1_score_index], prec[max_f1_score_index], c='red', marker='o')
    plt.legend()
    plt.grid(alpha=0.8)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve for mineral prediction")
    plt.tight_layout()
    
    file_path = os.path.abspath(__file__)
    file_dir = os.path.dirname(file_path)
    path_save = os.path.join(file_dir, '/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/')

    plt.savefig(path_save+ mode + 'precision-recall' +'_'+str(index)+'.png', dpi=300)

    # plt.savefig('Bayesian_main/run/figs/precision-recall.png', dpi=300)

def get_PR_curve(pred_list, mode):
    plt.figure()
    for i in range(len(pred_list)):
        y_test_fold, y_arr = pred_list[i]
        plot_PR(y_test_fold, y_arr, i+1, mode)

def get_PR_curve2(pred_list, mode):
    plt.figure()
    for i in range(len(pred_list)):
        y_test_fold, y_arr = pred_list[i]
        plot_PR2(y_test_fold, y_arr, i+1, mode)
    
def get_ROC_curve(pred_list, mode):
    plt.figure()

    for i in range(len(pred_list)):
        y_test_fold, y_arr = pred_list[i]
        fpr, tpr, thersholds = roc_curve(y_test_fold, y_arr)
        plot_roc(fpr, tpr, i+1, mode)
    
def get_ROC_curve2(pred_list, mode):
    plt.figure()

    for i in range(len(pred_list)):
        y_test_fold, y_arr = pred_list[i]
        fpr, tpr, thersholds = roc_curve(y_test_fold, y_arr)
        plot_roc2(fpr, tpr, i+1, mode) 
        


def plot_roc2(fpr, tpr, scat=False, save=True, mode='conformal'):
    """
        plot ROC curve
    """
    roc_auc = auc(fpr, tpr)
    # plt.plot(fpr, tpr, label=f'split set {index}, ROC area = {roc_auc:.2f}', lw=2)
    plt.plot(fpr, tpr, label=f' ROC area = {roc_auc:.2f}', lw=2)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    if scat:
        plt.scatter(fpr, tpr)
    plt.title("ROC curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

    if save:
        print("saving")
        plt.grid(alpha=0.8)
        plt.legend()
        plt.tight_layout()
        # Save the figure
        file_path = os.path.abspath(__file__)
        file_dir = os.path.dirname(file_path)
        path_save = os.path.join(file_dir, '/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/')

        plt.savefig(path_save + mode + 'roc' + '.png', dpi=300)
        # plt.savefig(f'Bayesian_main/run/figs/roc.png')
    else:
        plt.show()    


def get_confusion_matrix(cfm_list, clusters):
    """
    plot the confusion matrix
    """
    cols = clusters
    fig, axes = plt.subplots(nrows=1, ncols=cols, figsize=(10, 5))
    for i, plt_image in enumerate(cfm_list):
        index2 = i 
        axes[index2].matshow(plt_image, cmap=plt.get_cmap('Blues'), alpha=0.5)
        axes[index2].set_title(f"split set {i+1}")

        # Add labels to each cell
        for j in range(plt_image.shape[0]):
            for k in range(plt_image.shape[1]):
                text = plt_image[j, k]
                axes[index2].annotate(text, xy=(k, j), ha='center', va='center', 
                                      color='black',  weight='heavy', 
                                      bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', lw=1.5, alpha=0.8))
                
                # Add labels for y_pred
                axes[index2].set_xticks(np.arange(plt_image.shape[1]))
                axes[index2].set_xticklabels(np.arange(plt_image.shape[1]))
                axes[index2].set_xlabel("Prediction")

                # Add labels for y_true
                axes[index2].set_yticks(np.arange(plt_image.shape[0]))
                axes[index2].set_yticklabels(np.arange(plt_image.shape[0]))
                axes[index2].set_ylabel("Label")

    fig.tight_layout()
    
    
    plt.savefig('./Bayesian_main/run/figs/cfm.png')


def plot_split_standard(common_mask, label_arr, test_mask, save_path=None):
    """
        Plot to demonstrate data split
    """
    plt.figure(dpi=300)
    x, y = common_mask.nonzero()
    positive_x = x[label_arr.astype(bool)]
    positive_y = y[label_arr.astype(bool)]
    test_x, test_y = test_mask.nonzero()
    plt.scatter(x, y)

    plt.scatter(test_x, test_y, color='red')
    plt.scatter(positive_x, positive_y, color='gold')
    plt.legend(['Valid Region', 'Test-set', 'Positive samples'])
    
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.savefig('./run/spilt_standard.png')






import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os
from scipy.ndimage import median_filter

def image_conformal(result_values, mask, deposit_mask, test_mask=None, index=0, clusters=4, name='result_map', filter=False):
    plt.figure(dpi=600)
    plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=False)

    validYArray, validXArray = np.where(mask > 0)
    dep_YArray, dep_XArray = np.where(np.logical_and(deposit_mask, test_mask) == 1)
    result_array = np.zeros_like(deposit_mask, dtype="float")

    for i in range(len(validYArray)):
        if test_mask[validYArray[i], validXArray[i]] > 0:
            result_array[validYArray[i], validXArray[i]] = result_values[i] 

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
    file_path = os.path.abspath(__file__)
    file_dir = os.path.dirname(file_path)
    path_save = os.path.join(file_dir, '/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/')

    plt.savefig(path_save+name+'_'+str(index)+'.png', dpi=300)
    
"""
    
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
            
    # val1 = np.quantile(scores1,np.ceil((N1+1)*(1-alpha1))/N1)


    N2 = softmax_outputs_neg.shape[0]
    scores2=np.zeros(N2)

    for i in range(N2):
        if y_val_fold[i] == 0:
            scores2[i]= 1-softmax_outputs_neg[i]
        #else:
        #    scores2[i] = softmax_outputs_neg[i]
        
    non_zero_count2 = np.count_nonzero(scores2)
        
    # val2 = np.quantile(scores2,np.ceil((N2+1)*(1-alpha2))/N2)
    
    val1 = np.quantile(scores1,np.ceil((non_zero_count1+1)*(1-alpha1))/non_zero_count1)
    val2 = np.quantile(scores2,np.ceil((non_zero_count2+1)*(1-alpha2))/non_zero_count2)     
    
    return val1, val2
"""


def calculate_weights(w, x):
    """
    Calculate the weights for each point in the data set and for the test point x.

    :param w: Array of weights for the data points.
    :param x: Weight for the test point.
    :return: Array of normalized weights and the normalized weight for the test point.
    """
    w_sum = np.sum(w) + x
    return w / w_sum, x / w_sum

def calculate_weighted_quantile(scores, weights, alpha):
    """
    Calculate the weighted quantile for the given scores and weights.

    :param scores: Array of nonconformity scores.
    :param weights: Array of weights corresponding to the scores.
    :param alpha: Miscoverage level.
    :return: The weighted quantile.
    """
    # Sort scores and weights together
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Calculate the cumulative sum of the weights
    cum_weights = np.cumsum(sorted_weights)
    print("cum_weights", cum_weights)
    
    # Find the smallest score such that the cumulative sum of weights is >= 1 - alpha
    threshold_index = np.where(cum_weights >= 1 - alpha)[0][0]
    return sorted_scores[threshold_index]

def weighted_qhat(algo, X_val_fold, y_val_fold, alpha1, alpha2, w_train, w_test):
    """
    Calculate the weighted quantiles for the positive and negative classes.

    :param algo: Trained algorithm capable of producing probability predictions.
    :param X_val_fold: Validation features.
    :param y_val_fold: Validation labels.
    :param alpha1: Miscoverage level for the positive class.
    :param alpha2: Miscoverage level for the negative class.
    :param w_train: Array of weights for the training data points.
    :param w_test: Weight for the test point.
    :return: Weighted quantiles for the positive and negative classes.
    """
    probabilities = algo.predict_proba(X_val_fold)
    
    # Split the probabilities for positive and negative classes
    prob_positive_class = probabilities[:, 1]
    prob_negative_class = probabilities[:, 0]

    # Calculate nonconformity scores
    scores1 = (y_val_fold == 1) * (1 - prob_positive_class)
    scores2 = (y_val_fold == 0) * (1 - prob_negative_class)
    
    # Calculate weighted quantiles
    val1 = calculate_weighted_quantile(scores1, w_train, alpha1)
    # val2 = calculate_weighted_quantile(scores2, w_test, alpha2)
    val2 = 0.1
    return val1, val2
    
def get_qhat(algo, X_val_fold, y_val_fold, alpha):
    probabilities = algo.predict_proba(X_val_fold)
    softmax_outputs_pos = probabilities[:,1]
    softmax_outputs_neg = probabilities[:,0]
    
    N1 = softmax_outputs_pos.shape[0]
    scores1=np.zeros(N1)

    for i in range(N1):
        if y_val_fold[i] == 1:
            scores1[i]= 1-softmax_outputs_pos[i]
        else:
            scores1[i] = softmax_outputs_pos[i]
        
    non_zero_count1 = np.count_nonzero(scores1)
            
    # val1 = np.quantile(scores1,np.ceil((N1+1)*(1-alpha1))/N1)


    N2 = softmax_outputs_neg.shape[0]
    scores2=np.zeros(N2)

    for i in range(N2):
        if y_val_fold[i] == 0:
            scores2[i]= 1-softmax_outputs_neg[i]
        #else:
        #    scores2[i] = softmax_outputs_neg[i]
        
    non_zero_count2 = np.count_nonzero(scores2)
        
    
    val1 = np.quantile(scores1,np.ceil((N1+1)*(1-alpha))/N1)   
    
    return val1


def get_qhat1(algo, X_val_fold, y_val_fold, alpha1):
    probabilities = algo.predict_proba(X_val_fold)
    softmax_outputs_pos = probabilities[:,1]
    # print(softmax_outputs_pos)
    
    N1 = softmax_outputs_pos.shape[0]
    scores1=np.zeros(N1) 

    for i in range(N1):
        if y_val_fold[i] == 1:
            scores1[i]= 1-softmax_outputs_pos[i]
        #else:
        #    scores1[i] = softmax_outputs_pos[i]
        
    # print(scores1)
        
    non_zero_count = np.count_nonzero(scores1)
    # print("Number of non-zero values in scores1:", non_zero_count)
    
    quantile_value = np.ceil((non_zero_count+1)*(1-alpha1))/non_zero_count
    # print(alpha1)
    # print(quantile_value)
            
    # val1 = np.quantile(scores1,np.ceil((N1+1)*(1-alpha1))/N1)
    val1 = np.quantile(scores1, quantile_value)

    return val1

def get_qhat2(algo, X_val_fold, y_val_fold, alpha2):
    probabilities = algo.predict_proba(X_val_fold)
    softmax_outputs_neg = probabilities[:,0]
    softmax_outputs_pos = probabilities[:,1]
    
    N1 = softmax_outputs_pos.shape[0]
    scores1=np.zeros(N1) 

    for i in range(N1):
        if y_val_fold[i] == 1:
            scores1[i]= 1-softmax_outputs_pos[i]
        else:
             scores1[i] = softmax_outputs_pos[i]

    
    quantile_value = (1-alpha2) * ((N1 + 1) / N1)
    val1 = np.percentile(scores1, quantile_value*100)


    N2 = softmax_outputs_neg.shape[0]
    scores2=np.zeros(N2)

    for i in range(N2):
        if y_val_fold[i] == 0:
            scores2[i]= 1-softmax_outputs_neg[i]
        else:
            scores2[i] = softmax_outputs_neg[i]
        
    val2 = np.percentile(scores2, quantile_value*100) 
    
    # assert val2>val1
    
    return val1, val2, probabilities


def new_qhat(algo, X_val_fold, y_val_fold, alpha2):
    probabilities = algo.predict_proba(X_val_fold)
    softmax_outputs_neg = probabilities[:,0]
    softmax_outputs_pos = probabilities[:,1]

    N2 = softmax_outputs_neg.shape[0]
    scores2=np.zeros(N2)

    for i in range(N2):
        if y_val_fold[i] == 0:
            scores2[i]= 1-softmax_outputs_neg[i]
        #else:
        #    scores2[i] = softmax_outputs_neg[i]
            

    # val2 = np.quantile(scores2,np.ceil((N2+1)*(1-alpha2))/N2)     
    val2 = np.quantile(scores2,np.ceil((non_zero_count+1)*(1-alpha2))/non_zero_count)    
    
    return val2


def get_coverage(y_val, conf_sets):
    total_data_points = len(y_val)
    conformal_data_points_pos = 0
    conformal_data_points_neg = 0


    for i in range(total_data_points):
    # Check if the true label is in the prediction set (interval)
        if y_val[i] == (conf_sets[i]=='1'):
            conformal_data_points_pos += 1
        elif y_val[i] == (conf_sets[i]=='0'):
            conformal_data_points_neg += 1
            
    # Calculate coverage as the proportion of conformal data points
    coverage_pos = conformal_data_points_pos / total_data_points
    coverage_neg = conformal_data_points_neg / total_data_points
    
    return coverage_pos, coverage_neg

def get_coverage_2(y_val, conf_sets):
    total_data_points = len(y_val)
    conformal_data_points = 0
    for i in range(total_data_points):
        if y_val[i] == conf_sets[i] :
            conformal_data_points += 1

    # Calculate overall coverage
    coverage = conformal_data_points / total_data_points

    return coverage

def evaluate_conformal_prediction(algo, X_test, y_test, alpha):
    conformal_set = get_conformal_set(algo, X_test, alpha)
    conformal_values = get_conformal_values(conformal_set)
    auc_score = roc_auc_score(y_test, conformal_values)
    return auc_score

def get_conformal_set(probabilities, feature, val1, val2 ):
    # probabilities = algo.predict_proba(feature)

    # Split the probabilities for positive and negative classes
    prob_positive_class = probabilities[:, 1]
    prob_negative_class = probabilities[:, 0]

    N = len(feature)
    # print("prob_positive_class", len(prob_positive_class))
    # print("prob_negative_class", len(prob_negative_class))
    conf_sets = []
    for i in range(N):
        if prob_positive_class[i] >= 1 - val1: #changed self.q_yhat_pos to val
            conf_sets.append('1')  # Predicted as positive
        else:
            conf_sets.append('(0,1)')  # Predicted as not sure
            
    for i in range(N):
        if prob_negative_class[i] >= 1 - val2:
            conf_sets[i] = '0' 
            
    return conf_sets

def get_conformal_set_withprob(algo, feature, val1, val2 ):
    probabilities = algo.predict_proba(feature)

    # Split the probabilities for positive and negative classes
    prob_positive_class = probabilities[:, 1]
    prob_negative_class = probabilities[:, 0]
    

    N = len(feature)
    conf_sets = []
    for i in range(N):
        if prob_positive_class[i] >= 1 - val1:
            conf_sets.append('1')  # Predicted as positive
        else:
            conf_sets.append('(0,1)')  # Predicted as not sure
            
    for i in range(N):
        if prob_negative_class[i] >= 1 - val2:
            conf_sets[i] = '0' 
            
    return conf_sets

def get_conformal_set_false(algo, feature, val1 = None):
    probabilities = algo.predict_proba(feature)

    # Split the probabilities for positive and negative classes
    prob_positive_class = probabilities[:, 0]

    N = len(feature)
    # print("prob_positive_class", len(prob_positive_class))
    # print("prob_negative_class", len(prob_negative_class))
    conf_sets = []
    for i in range(N):
        if prob_positive_class[i] >= 1 - val1: #changed self.q_yhat_pos to val
            conf_sets.append('0')  # Predicted as positive
        else:
            conf_sets.append('1' )  # Predicted as not sure
            
    return conf_sets

def calculate_accuracy_uncertain_incorrect(conformal_values, y_true):

    # Convert uncertain predictions (0.5) to incorrect predictions
    # Here, we consider uncertain predictions as incorrect
    definite_predictions = np.where(conformal_values == 0.5, 1 - y_true, conformal_values)

    # Count the number of correct predictions
    correct_predictions = np.sum(definite_predictions == y_true)

    # Calculate accuracy
    accuracy = correct_predictions / len(y_true)
    
    return accuracy

"""def get_conformal_set_withOOD(algo, feature, val1, val2):
    probabilities = algo.predict_proba(feature)
    prob_positive_class = probabilities[:, 1]
    prob_negative_class = probabilities[:, 0]

    N = len(feature)
    conf_sets = []

    for i in range(N):
        if prob_positive_class[i] >= 1 - val1 and prob_negative_class[i] >= 1 - val2:
            conf_sets.append('OOD')  # Flag as potential OOD instance
        elif prob_positive_class[i] >= 1 - val1:
            conf_sets.append('1')  # Predicted as positive
        elif prob_negative_class[i] >= 1 - val2:
            conf_sets.append('0')  # Predicted as negative
        else:
            conf_sets.append('(0,1)')  # Not sure
            
    return conf_sets"""

"""def get_conformal_set(algo, feature, val):
    probabilities = algo.predict_proba(feature)
    conf_sets = []

    # Adaptive thresholding based on probability distribution
    # Could further refine based on performance metrics or domain knowledge
    for prob_pos, prob_neg in probabilities:
        if prob_pos >= 1 - val:
            conf_sets.append(1)  # Positive
        elif prob_neg >= 1 - val:
            conf_sets.append(0)  # Negative
        else:
            conf_sets.append(None)  # Ambiguous or OOD

    return conf_sets"""

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



def calculate_accuracy(conformal_values, y_true):
    correct_predictions = np.sum((conformal_values == 1) & (y_true == 1)) + np.sum((conformal_values == 0) & (y_true == 0))
    accuracy = correct_predictions / len(y_true)
    
    return accuracy

def new_calculate_accuracy(conformal_values, y_true):
    # print(len(y_true))
    truecount_0 = np.count_nonzero(y_true == 0)
    truecount_1 = np.count_nonzero(y_true == 1)
    # print(truecount_0)
    # print(truecount_1)
    pos_prediction = np.sum((conformal_values == 1) & (y_true == 1))  
    neg_prediction = np.sum((conformal_values == 0) & (y_true == 0))
    pos_accuracy = pos_prediction / truecount_1
    neg_accuracy = neg_prediction / truecount_0
    
    return pos_accuracy, neg_accuracy

def show_result_map_withdeposit(result_values, mask, deposit_mask, test_mask=None, index=0, clusters=4, name='result_map', filter=False):
    plt.figure(dpi=600)
    plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=False)

    validYArray, validXArray = np.where(mask > 0)
    dep_YArray, dep_XArray = np.where(deposit_mask > 0)
    result_array = np.zeros_like(deposit_mask, dtype="float")

    for i in range(len(validYArray)):
        if test_mask is None or test_mask[validYArray[i], validXArray[i]] > 0:
            result_array[validYArray[i], validXArray[i]] = result_values[i] * 100

    if filter:
        result_array = median_filter(result_array, size=3)

    result_array[~mask] = np.nan

    plt.imshow(result_array, cmap='viridis')
    plt.scatter(dep_XArray, dep_YArray, c='red', s=40, alpha=0.5, label='Deposit Mask')  # Overlay deposit mask

    cbar = plt.colorbar(shrink=0.75, aspect=30, pad=0.06)
    cbar.ax.set_ylabel('Prediction', fontsize=25)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper left', fontsize=18)
    plt.tight_layout()
    plt.gca().set_facecolor('lightgray')

    file_path = os.path.abspath(__file__)
    file_dir = os.path.dirname(file_path)
    path_save = os.path.join(file_dir, '/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/')

    plt.savefig(path_save+name+'_'+str(index)+'.png', dpi=300)
    


def show_result_map_withdeposit_for_fold(result_values, mask, deposit_mask, test_mask=None, index=0, name='result_map', filter=False):
    plt.figure(dpi=600)
    plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=False)

    validYArray, validXArray = np.where(mask > 0)
    dep_YArray, dep_XArray = np.where(deposit_mask > 0)
    result_array = np.zeros_like(deposit_mask, dtype="float")

    for i in range(len(validYArray)):
        if test_mask is None or test_mask[validYArray[i], validXArray[i]] > 0:
            result_array[validYArray[i], validXArray[i]] = result_values[i] * 100

    if filter:
        result_array = median_filter(result_array, size=3)

    result_array[~mask] = np.nan

    plt.imshow(result_array, cmap='viridis')

    # Plot only the deposit points that are within the test mask
    if test_mask is not None:
        dep_masked_YArray, dep_masked_XArray = np.where((deposit_mask > 0) & (test_mask > 0))
        plt.scatter(dep_masked_XArray, dep_masked_YArray, c='red', s=40, alpha=0.5, label='Deposit Mask')  
    else:
        plt.scatter(dep_XArray, dep_YArray, c='red', s=20, alpha=0.5, label='Deposit Mask')  

    cbar = plt.colorbar(shrink=0.75, aspect=30, pad=0.06)
    cbar.ax.set_ylabel('Prediction', fontsize=25)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper left', fontsize=18)
    plt.tight_layout()
    plt.gca().set_facecolor('lightgray')

    file_path = os.path.abspath(__file__)
    file_dir = os.path.dirname(file_path)
    path_save = os.path.join(file_dir, '/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/')

    plt.savefig(path_save+name+'_'+str(index)+'.png', dpi=300)


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
            result_array[validYArray[i], validXArray[i]] = result_values[i] 

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


    # Save the figure
    file_path = os.path.abspath(__file__)
    file_dir = os.path.dirname(file_path)
    path_save = os.path.join(file_dir, '/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/')

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
            result_array[validYArray[i], validXArray[i]] = result_values[i] 

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
    file_path = os.path.abspath(__file__)
    file_dir = os.path.dirname(file_path)
    path_save = os.path.join(file_dir, '/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/')

    plt.savefig(path_save+name+'_'+str(index)+'.png', dpi=300)
    
def show_result_map(result_values, mask, deposit_mask, test_mask=None, index=0, clusters=4, name='result_map', filter=False):

    plt.figure(dpi=600)
    plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=False)

    validYArray, validXArray = np.where(mask > 0)
    dep_YArray, dep_XArray = np.where(np.logical_and(deposit_mask, test_mask) == 1)
    result_array = np.zeros_like(deposit_mask, dtype="float")
    
    plt.scatter(dep_XArray, dep_YArray, c='red', s=20, alpha=0.5, label='Deposit Mask')  # Overlay deposit mask

    for i in range(len(validYArray)):
        if test_mask[validYArray[i], validXArray[i]] > 0:
            result_array[validYArray[i], validXArray[i]] = result_values[i] * 100

    if filter:
        result_array = median_filter(result_array, size=3)
   

    result_array[~mask] = np.nan

    plt.imshow(result_array, cmap='viridis')
    plt.rcParams['font.size'] = 18

    # Plot target points with improved style
    # plt.scatter(dep_XArray, dep_YArray, c='red', s=20, alpha=0.5, )


    # Add a legend
    # plt.legend(loc='upper left',fontsize=18)
    cbar = plt.colorbar(shrink=0.75, aspect=30, pad=0.06)
    cbar.ax.set_ylabel('Prediction', fontsize=25)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)

    # Adjust subplot spacing
    plt.tight_layout()
    plt.gca().set_facecolor('lightgray')

    # Save the figure
    file_path = os.path.abspath(__file__)
    file_dir = os.path.dirname(file_path)
    path_save = os.path.join(file_dir, '/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/')

    plt.savefig(path_save+name+'_'+str(index)+'.png', dpi=300)

def show_result_map_with_deposit(result_values, mask, deposit_mask, test_mask=None, index=0, clusters=4, name='result_map', filter=False):

    plt.figure(dpi=600)
    plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=False)

    validYArray, validXArray = np.where(mask > 0)
    dep_YArray, dep_XArray = np.where(np.logical_and(deposit_mask, test_mask) == 1)
    result_array = np.zeros_like(deposit_mask, dtype="float")

    for i in range(len(validYArray)):
        if test_mask[validYArray[i], validXArray[i]] > 0:
            result_array[validYArray[i], validXArray[i]] = result_values[i] * 100

    if filter:
        result_array = median_filter(result_array, size=3)
   

    result_array[~mask] = np.nan

    plt.imshow(result_array, cmap='viridis')
    plt.rcParams['font.size'] = 18

    # Plot target points with improved style
    # plt.scatter(dep_XArray, dep_YArray, c='red', s=20, alpha=0.5, )


    # Add a legend
    # plt.legend(loc='upper left',fontsize=18)
    cbar = plt.colorbar(shrink=0.75, aspect=30, pad=0.06)
    cbar.ax.set_ylabel('Prediction', fontsize=25)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)

    # Adjust subplot spacing
    plt.tight_layout()
    plt.gca().set_facecolor('lightgray')

    # Save the figure
    file_path = os.path.abspath(__file__)
    file_dir = os.path.dirname(file_path)
    path_save = os.path.join(file_dir, '/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/')

    plt.savefig(path_save+name+'_'+str(index)+'.png', dpi=300)


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
    return

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

def add_mask_new(test_mask, common_mask):
    new_mask = np.zeros_like(test_mask[0])
    # 对test_mask[0]、test_mask[1]、test_mask[2]、test_mask[3]进行或操作
    for i in range(len(test_mask)):
        new_mask = new_mask | test_mask[i]
    new_mask = ~new_mask
    new_mask = new_mask * common_mask
    # plt.imshow(new_mask)
    test_mask = np.concatenate((test_mask,np.expand_dims(new_mask, axis=0)),axis=0)
    return test_mask

def add_mask(test_mask):
    new_mask = np.zeros_like(test_mask[0])
    # 对test_mask[0]、test_mask[1]、test_mask[2]、test_mask[3]进行或操作
    for i in range(len(test_mask)):
        new_mask = new_mask | test_mask[i]
    new_mask = ~new_mask
    test_mask = np.concatenate((test_mask,np.expand_dims(new_mask, axis=0)),axis=0)
    return test_mask

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

def load_test_mask(name):
    if 'ag' in name.lower():
        return np.load('./temp/Ag_mask.npy')

    if 'cu' in name.lower():
        return np.load('./temp/Cu_mask.npy')
    
    if 'nova' in name.lower():
        return np.load('./temp/Au_mask.npy')
    
    return

def autoMPM(data_dir, run_mode = 'IID', optimize_step = 40, metrics=['auc', 'f1', 'pre']):
    
    if run_mode == 'IID':
        mode = 'random'
    else:
        mode  = 'k_split'

    path_list = os.listdir(data_dir), 
    for name in path_list:
        path = data_dir + '/' + name

        """# Automatically decide an algorithm
        algo_list = [rfcAlgo, svmAlgo, logiAlgo, NNAlgo]
        method = Method_select(algo_list)
        score = method.select(data_path=path, task=Model, mode=mode)
        algo = algo_list[score.index(max(score))]
        print("Use" + str(algo)) """
        
        algo = rfBoostAlgo
        
        # Bayesian optimization process
        bo = Bayesian_optimization(
            data_path=path, 
            algorithm=algo, 
            mode=mode,
            metrics=['auc', 'f1', 'pre'],
            default_params= True
            )
        
        x_best = bo.optimize(steps=optimize_step)



def get_feature_importance_by_shap(file_path, per=15):
    '''
    Get the feature importance by shap values
    file_path: the location where the shap_values are saved
    per: the percentage of features to be filtered by shap values
    return: the indices of features to be selected
    '''
    shap_values = np.load(file_path, allow_pickle=True)
    shap_values = [shap_values[0], shap_values[1]]


     
    feature_importance = np.abs(shap_values).mean(0)
    
    threshold = np.percentile(feature_importance, per)
    selected_features = feature_importance > threshold
    top_per_indices = np.argsort(np.sum(selected_features, axis=0))[::-1][:int((1-per/100)*selected_features.shape[1])]
    return top_per_indices,shap_values

