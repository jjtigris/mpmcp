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
# Fix for deprecated np.bool
np.bool = np.bool_

data_path = 'G:/A PYTHON NOTEBOOK/Bayesian_main/ooddata/Washington_new.pkl'
with open(data_path, 'rb') as f:
    feature_arr, label_arr, common_mask, feature_name_list = pickle.load(f)
height, width = common_mask.shape
label_arr = label_arr.astype(int)
print(common_mask.shape)

print(feature_name_list)




# Load data
data_path2 = 'G:/A PYTHON NOTEBOOK/Bayesian_main/ooddata/Washington.pkl'

with open(data_path2, 'rb') as f:
    _, _, common_mask_old, deposite_mask_old = pickle.load(f)
    
print(common_mask_old.shape)

new_shape = (height,width)
common_mask_new = np.zeros(new_shape, dtype=common_mask_old.dtype)

# Copy the values from the original array to the new array
common_mask_new[:common_mask_old.shape[0], :common_mask_old.shape[1]] = common_mask_old

deposit_mask_new = np.zeros(new_shape, dtype=deposite_mask_old.dtype)

# Copy the values from the original array to the new array
deposit_mask_new[:deposite_mask_old.shape[0], :deposite_mask_old.shape[1]] = deposite_mask_old
test_clusters=4
    


from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
# scaler = MinMaxScaler()
# feature_arr[:,2:][feature_arr[:,2:]<1e-5] = 0
feature_arr = scaler.fit_transform(feature_arr)

x_arr, y_arr = common_mask.nonzero()
positive_x = x_arr[label_arr.astype(bool)].reshape((-1,1))
positive_y = y_arr[label_arr.astype(bool)].reshape((-1,1))

row_idx, col_idx = np.where(common_mask)

# 然后使用 row_idx 和 col_idx 来构建 label_arr2d
label_arr2d = np.zeros(common_mask.shape, dtype=int)
label_arr2d[row_idx, col_idx] = ~label_arr

fig, ax = plt.subplots(figsize=(10, 10))

# 使用 imshow 函数绘制 label_arr2d
ax.imshow(label_arr2d, cmap='binary')

ax.set_title('2D Visualization of Label Array')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.savefig("G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/label.png")
plt.close()


real_mask = common_mask
common_mask = np.ones_like(real_mask)

X_train, X_test, y_train, y_test = train_test_split(
    feature_arr, label_arr, test_size=0.2, random_state=42)

# Use mutual information for feature selection
selector = SelectKBest(mutual_info_classif, k=3)
selector.fit(X_train, y_train)
selected_features_mi = selector.get_support(indices=True)

# Use recursive feature elimination
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=3)
rfe.fit(X_train, y_train)
selected_features_rfe = rfe.get_support(indices=True)
    
# Determine the number of features
num_features = X_train.shape[1]

# Feature names
# feature_names = ["Ag", "As", "Be", "Ca", "Co", "Cr", "Cu", "Fe", "La", "Mg", "Mn", "Ni", "Pb", "Ti"]
# feature_names = ["As", "B", "Ca", "Co", "Cr", "Cu", "Fe", "Mg", "Mn", "Ni", "Pb", "Sc", "Y"]

feature_names = feature_name_list
# Iterate over each feature to compute the treatment effect
treatment_effects = {}
for treatment_idx in range(num_features):
    print(f"Evaluating treatment effect for feature index {treatment_idx}")
    
    # Specify the treatment feature
    T_train = X_train[:, treatment_idx]
    T_test = X_test[:, treatment_idx]
    
    # Remove the treatment feature from the features set
    X_train_t = np.delete(X_train, treatment_idx, axis=1)
    X_test_t = np.delete(X_test, treatment_idx, axis=1)
    
    # Train the Causal Forest to estimate the treatment effect
    causal_forest = CausalForestDML(model_y=LinearRegression(), model_t=LinearRegression(), discrete_treatment=False)
    causal_forest.fit(y_train, T_train, X=X_train_t)
    
    # Evaluate the treatment effect on the test set
    treatment_effect = causal_forest.effect(X_test_t)
    treatment_effects[treatment_idx] = treatment_effect

# Plot all treatment effects on a single plot
fig, ax = plt.subplots(figsize=(15, 10))

# Define a color map for different features
colors = plt.cm.get_cmap('tab10', num_features)

# Legends for mean and variance
mean_variance_legend = []

for treatment_idx, treatment_effect in treatment_effects.items():
    ax.hist(treatment_effect, bins=30, alpha=0.7, label=f'{treatment_idx}', color=colors(treatment_idx))
    mean_effect = np.mean(treatment_effect)
    variance_effect = np.var(treatment_effect)
    mean_variance_legend.append(f'{feature_name_list[treatment_idx]}: Mean={mean_effect:.4f}, Var={variance_effect:.4f}')

ax.set_xlabel('Estimated Treatment Effect')
ax.set_ylabel('Frequency')
ax.set_title('Treatment Effects of All Features')

# Primary legend for features
feature_legend = ax.legend(loc='upper right', fontsize=14)
ax.add_artist(feature_legend)

# Secondary legend for mean and variance
lines = [plt.Line2D([0], [0], color=colors(i), lw=4) for i in range(num_features)]
labels = mean_variance_legend
plt.legend(lines, labels, loc='upper left', fontsize=14, bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig('G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/compiled_treatment_effects_with_mean_variance.png', dpi=300)

# Save treatment effects
np.save('G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/treatment_effects.npy', treatment_effects)
treatment_effects = np.load('G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/treatment_effects.npy', allow_pickle=True).item()

# Calculate mean and variance for each feature
mean_effects = []
variance_effects = []
for feature_idx, effects in treatment_effects.items():
    mean_effect = np.mean(effects)
    variance_effect = np.var(effects)
    mean_effects.append(mean_effect)
    variance_effects.append(variance_effect)
    print(f"Feature {feature_idx}: Mean Effect = {mean_effect}, Variance = {variance_effect}")

# Convert to numpy arrays for statistical calculations
mean_effects = np.array(mean_effects)
variance_effects = np.array(variance_effects)

# Calculate overall mean and standard deviation
mean_of_means = np.mean(mean_effects)
std_of_means = np.std(mean_effects)
mean_of_variances = np.mean(variance_effects)
std_of_variances = np.std(variance_effects)

# Set dynamic thresholds
mean_threshold = mean_of_means + std_of_means
variance_threshold = mean_of_variances + std_of_variances

print(f"Dynamic Mean Threshold: {mean_threshold}")
print(f"Dynamic Variance Threshold: {variance_threshold}")

# Filter features based on dynamic thresholds
important_features = []
for feature_idx, (mean_effect, variance_effect) in enumerate(zip(mean_effects, variance_effects)):
    if abs(mean_effect) > mean_threshold or variance_effect > variance_threshold:
        important_features.append(feature_idx)

print(f"Selected Important Features: {important_features}")

# Combine selected features from causal analysis, mutual information, and RFE
causal_selected_features = important_features  # From causal analysis
all_selected_features = np.unique(np.concatenate([selected_features_mi, selected_features_rfe, causal_selected_features]))
print("Combined Selected Features:", all_selected_features)

# Use list comprehension to select the features based on the indices
selected_features = [feature_name_list[i] for i in all_selected_features]
print(feature_name_list)
print(selected_features)

# all_selected_features = [0,  5,  6,  7,  8,  9, 13, 15, 17, 19, 20] 
# all_selected_features = [ 0, 1,  2,  3,  5,  6,  7,  8,  9, 10, 11, 12, 15]
# Create a refined feature set

feature_arr = feature_arr[:, all_selected_features]

def km(x, y, cluster):
    """Clustering the positive samples with k-means

    Returns:
        array: The cluster id that each sample belongs to
    """
    coord = np.concatenate([x, y], axis=1)
    cl = KMeans(n_clusters=cluster, random_state=0).fit(coord)
    cll = cl.labels_
    
    return cll

def test_extend(x, y, test_num):
    """Extend from the start point to generate the tesk mask

    Args:
        x (int): The x coord of the start point to extend from.
        y (int): The y coord of the start point to extend from.
        test_num (_type_): The size of test set

    Returns:
        Array: Mask for test set
    """
    # Build the test mask
    test_mask = np.zeros_like(common_mask).astype(bool)
    test_mask[x, y] = True

    candidate = set([])
    for i in range(test_num-1):
        # Add the neighbor grid which is in the valid region and not chosen yet into the candidate set
        if x >= 1 and common_mask[x-1, y] and not test_mask[x-1, y]:
            candidate.add((x-1, y))
        if y >= 1 and common_mask[x, y-1] and not test_mask[x, y-1]:
            candidate.add((x, y-1))
        if x <= height-2 and common_mask[x+1, y] and not test_mask[x+1, y]:
            candidate.add((x+1, y))
        if y <= width-2 and common_mask[x, y+1] and not test_mask[x, y+1]:
            candidate.add((x, y+1))
        
        # Randomly choose the next grid to put in the test set
        pick = np.random.randint(0, len(candidate))
        x, y = list(candidate)[pick]
        candidate.remove((x,y))
        test_mask[x, y] = True
    return test_mask
        
"""def dataset_split(test_mask_list=None, modify=True):
    
    if test_mask_list is None:
        test_mask_list = []
        # Randomly choose the start grid
        
        mask_sum = real_mask.sum()
        test_num = int(mask_sum / test_clusters)
        x_arr, y_arr = real_mask.nonzero()
        positive_x = x_arr[label_arr.astype(bool)].reshape((-1,1))
        positive_y = y_arr[label_arr.astype(bool)].reshape((-1,1))
        cll = km(positive_x, positive_y, test_clusters)
        
        for i in range(test_clusters):
            cluster_arr = (cll == i)
            cluster_x = positive_x[cluster_arr].squeeze()
            cluster_y = positive_y[cluster_arr].squeeze()
            
            start = np.random.randint(0, cluster_arr.sum())
            x, y = cluster_x[start], cluster_y[start]
            test_mask = test_extend(x, y, test_num)
            test_mask_list.append(test_mask)
    else:
        for test_mask in test_mask_list:
            assert test_mask.shape == common_mask.shape
    # Buf the test mask
    tmpt = test_mask_list
    # Split the dataset
    dataset_list = []
    for test_mask in test_mask_list:
        train_mask = ~test_mask
        
        test_mask = test_mask & real_mask
        test_pos = np.where(test_mask)
        test_pos = np.array(test_pos)
        
        test_mask = test_mask[real_mask]
        
        train_mask = train_mask & real_mask
        train_pos = np.where(train_mask)
        train_pos = np.array(train_pos)

        train_mask = train_mask[real_mask]
        X_train_fold, X_test_fold = feature_arr[train_mask], feature_arr[test_mask]
        y_train_fold, y_test_fold = label_arr[train_mask], label_arr[test_mask]
        
        # Modify y_test_fold
        if modify:
            true_num = y_test_fold.sum()
            index = np.arange(len(y_test_fold))
            true_test = index[y_test_fold == 1]
            false_test = np.random.permutation(index[y_test_fold == 0])[:true_num]
            test = np.concatenate([true_test, false_test])
            X_test_fold = X_test_fold[test]
            y_test_fold = y_test_fold[test]
            
   
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
            
        dataset = (X_train_fold, y_train_fold, X_test_fold, y_test_fold)
        dataset_list.append(dataset)
    
    return tmpt, dataset_list"""
    
def dataset_split(test_mask_list=None, modify=True):
    
    if test_mask_list is None:
        test_mask_list = []
        # Randomly choose the start grid
        
        mask_sum = real_mask.sum()
        test_num = int(mask_sum / test_clusters)
        x_arr, y_arr = real_mask.nonzero()
        positive_x = x_arr[label_arr.astype(bool)].reshape((-1,1))
        positive_y = y_arr[label_arr.astype(bool)].reshape((-1,1))
        cll = km(positive_x, positive_y, test_clusters)
        
        for i in range(test_clusters):
            cluster_arr = (cll == i)
            cluster_x = positive_x[cluster_arr].squeeze()
            cluster_y = positive_y[cluster_arr].squeeze()
            
            start = np.random.randint(0, cluster_arr.sum())
            x, y = cluster_x[start], cluster_y[start]
            test_mask = test_extend(x, y, test_num)
            test_mask_list.append(test_mask)
    else:
        for test_mask in test_mask_list:
            assert test_mask.shape == common_mask.shape
    # Buf the test mask
    tmpt = test_mask_list
    # Split the dataset
    dataset_list = []
    for test_mask in test_mask_list:
        train_mask = ~test_mask
        
        test_mask = test_mask & real_mask
        test_pos = np.where(test_mask)
        test_pos = np.array(test_pos)
        
        test_mask = test_mask[real_mask]
        
        train_mask = train_mask & real_mask
        
        mask_sum = train_mask.sum()
        val_num = int(mask_sum / 4)
        x_arr, y_arr = train_mask.nonzero()
        train_deposite_mask = train_mask & label_arr2d
        
        
        positive_indices = np.argwhere(train_deposite_mask)
        positive_x = np.array([positive_indices[:, 0]]).reshape(-1, 1)
        positive_y = np.array([positive_indices[:, 1]]).reshape(-1, 1)
        
        cll = km(positive_x, positive_y, test_clusters)
        
        retries = 0
        while retries < 1:
            cluster_arr = (cll == 0)
            cluster_x = np.atleast_1d(positive_x[cluster_arr].squeeze())
            cluster_y = np.atleast_1d(positive_y[cluster_arr].squeeze())

            # Check if the cluster is empty
            if cluster_x.size > 0 and cluster_y.size > 0:
                start = np.random.randint(0, cluster_arr.sum())
                x, y = cluster_x[start], cluster_y[start]
                val_mask = test_extend(x, y, val_num)
                train_mask = train_mask & ~val_mask
                
                val_pos = np.where(val_mask)
                val_pos = np.array(val_pos)
                
                val_mask = val_mask[real_mask]
                X_val_fold, y_val_fold = feature_arr[val_mask], label_arr[val_mask]
                break  # Exit the while loop if a valid cluster is found
            else:
                retries += 1
                print(f"Validation cluster is empty. Retrying... ({retries}/{3})")
        if retries == 3:
            print("Failed to find a non-empty validation cluster after {max_retries} retries. Skipping...")
         
        
        train_pos = np.where(train_mask)
        train_pos = np.array(train_pos)
        
        train_mask = train_mask[real_mask]
        
        X_train_fold, X_test_fold = feature_arr[train_mask], feature_arr[test_mask]
        y_train_fold, y_test_fold = label_arr[train_mask], label_arr[test_mask]
        
   
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





class ImprovedMineralProspectivityCNN(nn.Module):
    def __init__(self, input_length):
        super(ImprovedMineralProspectivityCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.norm4 = nn.BatchNorm1d(256)

        # Initialize a dummy input to calculate the output size after conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            conv_out_size = self._get_conv_output(dummy_input)
        
        self.fc1 = nn.Linear(conv_out_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def _get_conv_output(self, x):
        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = F.relu(self.norm4(self.conv4(x)))
        return x.numel()  # Total number of elements in the tensor

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = F.relu(self.norm4(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


    


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

"""def show_result_map(result_values, mask, deposit_mask, test_mask=None, index=0, clusters=4, name='result_map', filter=False):

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

    path_save = "G:/A PYTHON NOTEBOOK/Bayesian_main/Bayesian_main/code/run/figs/"

    plt.savefig(path_save+name+'_'+str(index)+'.png', dpi=300)"""
    
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

def align_results_with_positions(result_values, positions, mask_shape):
    aligned_result_array = np.zeros(mask_shape, dtype="float")
    for i, pos in enumerate(positions):
        if i < len(result_values):
            aligned_result_array[int(pos[0]), int(pos[1])] = result_values[i]
    return aligned_result_array

# Function to calculate accuracy
def binary_accuracy(y_pred, y_true):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_true).float().sum()
    acc = correct_results_sum / y_true.shape[0]
    acc = torch.round(acc * 100)
    return acc



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


def get_qhat(model, device, X_val_fold, y_val_fold, alpha1_start=0.05, alpha2_start=0.1):
    model.eval()
    X_val_fold = torch.tensor(X_val_fold).float().to(device)
    with torch.no_grad():
        probabilities = model(X_val_fold).cpu().numpy()

    # Check if log transformation is needed
    if np.max(probabilities) < 0.01:
        log_transform = True
        probabilities = np.log(probabilities + 1e-10)  # Apply log transformation
    else:
        log_transform = False

    softmax_outputs_pos = probabilities[:, 0]
    softmax_outputs_neg = 1 - probabilities[:, 0]

    N1 = softmax_outputs_pos.shape[0]
    scores1 = np.zeros(N1)

    counts = np.bincount(y_val_fold)
    print(f"Number of 0s in y_val_fold: {counts[0]}")
    print(f"Number of 1s in y_val_fold: {counts[1]}")

    for i in range(N1):
        if y_val_fold[i] == 1:
            scores1[i] = softmax_outputs_pos[i]

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

    N2 = softmax_outputs_neg.shape[0]
    scores2 = np.zeros(N2)

    for i in range(N2):
        if y_val_fold[i] == 0:
            scores2[i] = 1 - softmax_outputs_neg[i]

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


def get_conformal_set(probabilities, val1, val2):
    probabilities = np.array(probabilities)  
    prob_positive_class = probabilities
    prob_negative_class = 1 - probabilities

    N = len(probabilities)

    conf_sets = []
    for i in range(N):
        if prob_positive_class[i] >= val1:
            conf_sets.append('1')
        else:
            conf_sets.append('(0,1)')
            
    for i in range(N):
        if prob_negative_class[i] >= 1 - val2:
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

# Function to calculate predictive uncertainty
def calculate_predictive_uncertainty(model, data_loader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            output = model(data)
            probs = output.cpu().numpy().squeeze()
            all_probs.append(probs)
    all_probs = np.array(all_probs)
    
    # Calculate variance across different subsets
    uncertainty = np.var(all_probs, axis=0)
    
    return uncertainty

# Function to evaluate predictive uncertainty on OOD data
def evaluate_predictive_uncertainty(model, ood_data, batch_size=32):
    ood_dataset = TensorDataset(torch.tensor(ood_data).float())
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)
    
    uncertainty = calculate_predictive_uncertainty(model, ood_loader)
    
    return uncertainty

def find_best_threshold(y_true, y_probs):
    best_threshold = 0.5
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision_score(y_true, y_pred)
            best_recall = recall_score(y_true, y_pred)
    
    return best_threshold, best_f1, best_precision, best_recall

# Function to balance the classes
def balance_classes(X, y):
    # Separate classes
    X_majority = X[y == 0]
    y_majority = y[y == 0]
    X_minority = X[y == 1]
    y_minority = y[y == 1]
    
    # Upsample minority class
    X_minority_upsampled, y_minority_upsampled = resample(
        X_minority, y_minority,
        replace=True,
        n_samples=len(y_majority),
        random_state=42
    )
    
    # Combine majority class with upsampled minority class
    X_balanced = np.vstack((X_majority, X_minority_upsampled))
    y_balanced = np.hstack((y_majority, y_minority_upsampled))
    
    return X_balanced, y_balanced

def calculate_prediction_intervals(model, data_loader, confidence_level=0.95):
    model.eval()
    all_lower_bounds = []
    all_upper_bounds = []
    z = 1.96  # z-value for 95% confidence interval

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            output = model(data)
            probs = output.cpu().numpy().squeeze()
            if probs.ndim == 0:
                probs = np.expand_dims(probs, axis=0)  # Ensure probs is always iterable

            interval_width = z * np.sqrt(probs * (1 - probs))
            lower_bound = np.clip(probs - interval_width, 0, 1)  # Ensure bounds are within [0, 1]
            upper_bound = np.clip(probs + interval_width, 0, 1)  # Ensure bounds are within [0, 1]

            all_lower_bounds.extend(lower_bound)
            all_upper_bounds.extend(upper_bound)
    
    lower_bounds = np.array(all_lower_bounds)
    upper_bounds = np.array(all_upper_bounds)
    return np.column_stack((lower_bounds, upper_bounds))

# Function to calculate coverage width
def calculate_coverage_width(prediction_intervals):
    lower_bounds, upper_bounds = prediction_intervals[:, 0], prediction_intervals[:, 1]
    coverage_widths = upper_bounds - lower_bounds
    average_coverage_width = np.mean(coverage_widths)
    return average_coverage_width

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

def calculate_metrics(labels, outputs):
    if len(np.unique(labels)) > 1:
        auc_score = roc_auc_score(labels, outputs)
        f1 = f1_score(labels, np.round(outputs))
    else:
        auc_score = None
        f1 = f1_score(labels, np.round(outputs))
    return auc_score, f1

test_mask_list, dataset_list = dataset_split()


input_length = feature_arr.shape[1]
model = ImprovedMineralProspectivityCNN(input_length)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Loss Function and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
# Number of epochs to train the model


early_stop_patience = 5
early_stop_counter = 0
best_val_loss = float('inf')

i=0
y_arr_record = []
conformal_values_list = []
num_epochs = 10
print(feature_arr.shape)

        

all_pos = np.where(real_mask)
all_pos = np.array(all_pos)
all_pos = np.transpose(all_pos)
new_feature_arr = np.concatenate([feature_arr, all_pos], axis=1)
        
full_dataset = TensorDataset(torch.tensor(new_feature_arr[:,:-2]).float(), torch.tensor(label_arr).long())
print(len(label_arr))
full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
test_outputs_total = []
test_labels_total = []
for dataset in dataset_list:
    X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_fold, y_test_fold= dataset
    
    print("X_train_fold", X_train_fold[:,:-2].shape)
    print("X_val_fold", X_val_fold.shape)
    print("X_test_fold", X_test_fold[:,:-2].shape)
    
    show_dataset_map(X_train_fold,test_mask_list,i+1,'train')
    show_dataset_map(X_test_fold,test_mask_list,i+1,'test')
    
    # X_train_fold, y_train_fold = balance_classes(X_train_fold, y_train_fold)
    
    _train_mask, _test_mask= get_mask(X_train_fold, X_test_fold, test_mask_list)
    
    print(f"{i+1}-fold")
  
    train_dataset = TensorDataset(torch.tensor(X_train_fold[:,:-2]).float(), torch.tensor(y_train_fold).long())
    val_dataset = TensorDataset(torch.tensor(X_val_fold).float(), torch.tensor(y_val_fold).long())
    test_dataset = TensorDataset(torch.tensor(X_test_fold[:,:-2]).float(), torch.tensor(y_test_fold).long())
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Training and Validation Loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_outputs = []
        train_labels = []
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data).squeeze()
            
                # Ensure output and target are properly shaped
            if output.dim() == 0:
                output = output.unsqueeze(0)
            if target.dim() == 0:
                target = target.unsqueeze(0)
            
            loss = criterion(output, target.float())
            acc = binary_accuracy(output, target)
            loss.backward()
            optimizer.step()
            train_outputs.extend(output.detach().cpu().numpy())
            train_labels.extend(target.detach().cpu().numpy())
            train_loss += loss.item()
            train_acc += acc.item()
        
        train_auc, train_f1 = calculate_metrics(train_labels, train_outputs)

        # Print training statistics
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc/len(train_loader):.2f}%,  Train AUC: {train_auc:.4f}, Train F1: {train_f1:.4f}')

         # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_outputs = []
        val_labels = []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data).squeeze()
            
                # Ensure output and target are properly shaped
                if output.dim() == 0:
                    output = output.unsqueeze(0)
                if target.dim() == 0:
                    target = target.unsqueeze(0)

                loss = criterion(output, target.float())
                acc = binary_accuracy(output, target)
                
                val_outputs.extend(output.detach().cpu().numpy())
                val_labels.extend(target.detach().cpu().numpy())
                val_loss += loss.item()
                val_acc += acc.item()
        val_auc, val_f1 = calculate_metrics(val_labels, val_outputs)


        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(val_loader):.4f}, Validation Acc: {val_acc/len(val_loader):.2f}%, Val AUC: {val_auc}, Val F1: {val_f1:.4f}')
    
    val1, val2 = get_qhat(model, device, X_val_fold, y_val_fold, alpha1_start=0.05, alpha2_start=0.05)
  # model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_outputs = []
    test_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data).squeeze()
        
            # Ensure output and target are properly shaped
            if output.dim() == 0:
                output = output.unsqueeze(0)
            if target.dim() == 0:
                target = target.unsqueeze(0)

            loss = criterion(output, target.float())
            acc = binary_accuracy(output, target)
            test_loss += loss.item()
            test_acc += acc.item()
            test_outputs.extend(output.detach().cpu().numpy())
            test_labels.extend(target.detach().cpu().numpy())
            test_outputs_total.extend(output.detach().cpu().numpy())
            test_labels_total.extend(target.detach().cpu().numpy())
            
    test_outputs = np.array(test_outputs)
    
    best_threshold, best_f1, best_precision, best_recall = find_best_threshold(test_labels, test_outputs)
    test_predictions = (test_outputs >= best_threshold).astype(int)
    test_f1 = f1_score(test_labels, test_predictions)
    test_auc = roc_auc_score(test_labels, test_outputs)
    
    plot_precision_recall(test_labels, test_outputs, index=i+1, name='PR_curve')
    plot_roc_curve(test_labels, test_outputs, index=i+1, name='roc_curve')
            
    print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_acc/len(test_loader):.2f}%, Test AUC: {test_auc:.4f}, Test F1: {test_f1:.4f}')
    print(f'Best Threshold: {best_threshold:.2f}, Best F1: {best_f1:.4f}, Best Precision: {best_precision:.4f}, Best Recall: {best_recall:.4f}')
    
    uncertainty = np.var(test_outputs, axis=0)
    # Print or save the predictive uncertainty values
    print("Predictive Uncertainty for OOD Data:", uncertainty)
    
    prediction_intervals = calculate_prediction_intervals(model, test_loader)
    average_coverage_width = calculate_coverage_width(prediction_intervals)
    print("average_coverage_width", average_coverage_width)


    tn, fp, fn, tp = confusion_matrix(test_labels, np.round(test_outputs)).ravel()
    tpr = tp / (tp + fn)  # True Positive Rate
    fpr = fp / (fp + tn)  # False Positive Rate

    print(f'Test TPR: {tpr:.4f}, Test FPR: {fpr:.4f}')
    
    conf_sets = get_conformal_set(np.array(test_outputs), val1, val2)
    conf_values = get_conformal_values(conf_sets)
    
    count_05 = np.count_nonzero(conf_values == 0.5)
    count_0 = np.count_nonzero(conf_values == 0.0)
    count_1 = np.count_nonzero(conf_values == 1.0)
    
    # print(len(test_conformal_values3))
    print("0.5", count_05)
    print("0", count_0)
    print("1", count_1)
    
    # Exclude uncertain predictions
    y_true_filtered3 = [test_labels[i] for i in range(len(test_labels)) if conf_values[i] != 0.5]
    y_pred_filtered3 = [pred for pred in conf_values if pred != 0.5]
    print("y_true_filtered3", len(y_true_filtered3))
    print(len(y_pred_filtered3))
    
    tn, fp, fn, tp = confusion_matrix(y_true_filtered3, y_pred_filtered3).ravel()
    tpr_after = tp / (tp + fn)
    fpr_after = fp / (fp + tn)
    print(f'conformal TPR: {tpr_after:.4f}, conformal FPR: {fpr_after:.4f}')
    test_auc2 = roc_auc_score(y_true_filtered3, y_pred_filtered3)
    test_f12 = f1_score(y_true_filtered3, y_pred_filtered3)
    print(f'conformal auc: {test_auc2:.4f}, conformal f1: {test_f12:.4f}')
    
    
    
    
    
    # Assuming the last two columns of X_test_fold are the positions
    test_positions = X_test_fold[:, -2:]
    
    # Generate test mask for visualization
    test_mask = np.zeros_like(real_mask, dtype=bool)
    for pos in test_positions:
        test_mask[int(pos[0]), int(pos[1])] = True
        
    full_outputs = []
    full_labels = []
    with torch.no_grad():
        for data, target in full_loader:
            data, target = data.to(device), target.to(device)

            output = model(data).squeeze()
        
            # Ensure output and target are properly shaped
            if output.dim() == 0:
                output = output.unsqueeze(0)
            if target.dim() == 0:
                target = target.unsqueeze(0)

            full_outputs.extend(output.detach().cpu().numpy())
            full_labels.extend(target.detach().cpu().numpy())
            

    all_positions = new_feature_arr[:, -2:]
    
    # Generate test mask for visualization
    all_mask = np.zeros_like(real_mask, dtype=bool)
    for pos in all_positions:
        all_mask[int(pos[0]), int(pos[1])] = True
            
    y_arr_feature_map = np.array(full_outputs)
    
    full_conf_sets = get_conformal_set(y_arr_feature_map, val1, val2)
    full_conf_values = get_conformal_values(full_conf_sets)
    
    y_true_filtered = [full_labels[z] for z in range(len(full_labels)) if full_conf_values[z] != 0.5]
    y_pred_filtered = [pred for pred in full_conf_values if pred != 0.5]
    tn, fp, fn, tp = confusion_matrix(y_true_filtered, y_pred_filtered).ravel()
    tpr_after = tp / (tp + fn)
    fpr_after = fp / (fp + tn)
    print(f'full conformal TPR: {tpr_after:.4f}, full conformal FPR: {fpr_after:.4f}')
    test_auc2 = roc_auc_score(y_true_filtered, y_pred_filtered)
    test_f12 = f1_score(y_true_filtered, y_pred_filtered)
    print(f'full conformal auc: {test_auc2:.4f}, full conformal f1: {test_f12:.4f}')
    

    
    aligned_full_results = align_results_with_positions(full_outputs, all_positions, real_mask.shape)
    y_arr_record.append(aligned_full_results)
    
    aligned_conformal = align_results_with_positions(list(full_conf_values), all_positions, real_mask.shape)
    conformal_values_list.append(aligned_conformal)
    
    
    show_result_map(result_values=aligned_full_results, mask=common_mask_new, deposit_mask=deposit_mask_new, test_mask=_test_mask, index=i+1, name='test_pred')
    show_result_map(result_values=aligned_full_results, mask=common_mask_new, deposit_mask=deposit_mask_new, test_mask=common_mask_new, index=i+1, name='all_pred')

    image_conformal_withdeposit_forfold(result_values=aligned_conformal, mask=common_mask_new, deposit_mask=deposit_mask_new, test_mask=_test_mask, index=i+1, name='conformal_test_pred')
     
    
    y_mask = test_mask_list[i].reshape(-1)
    y_mask = y_mask[common_mask.reshape(-1)]
    y_mask = y_mask.reshape(common_mask_new.shape)
    y_arr_record[i][~y_mask] = 0
    conformal_values_list[i][~y_mask] = 0

    
    i+=1


test_auc = roc_auc_score(test_labels_total, test_outputs_total)
test_f1 = f1_score(test_labels_total, np.round(test_outputs_total))
            
print(f'total test AUC: {test_auc:.4f}, test F1: {test_f1:.4f}')
    
tn, fp, fn, tp = confusion_matrix(test_labels_total, np.round(test_outputs_total)).ravel()
tpr = tp / (tp + fn)  # True Positive Rate
fpr = fp / (fp + tn)  # False Positive Rate

print(f'Test TPR: {tpr:.4f}, Test FPR: {fpr:.4f}')
    
plot_precision_recall(test_labels_total, test_outputs_total, index=6, name='PR_curve')
plot_roc_curve(test_labels_total, test_outputs_total, index=6, name='roc_curve')           
# Save the model after training
y_arr_record = np.maximum.reduce(y_arr_record)
show_result_map(result_values=y_arr_record, mask=common_mask_new, deposit_mask=deposit_mask_new, test_mask=common_mask_new, index=0, name='concatenate_all_pred',filter=True)

conformal_record_2 = np.maximum.reduce(conformal_values_list)
image_conformal_withdeposit(result_values=conformal_record_2, mask=common_mask_new, deposit_mask=deposit_mask_new, test_mask=common_mask_new, index=0, name='dynamic_all_conformal',filter=True)
   
   
print("Model saved successfully.")
