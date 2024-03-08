import pykrige
from scipy.interpolate import interp2d
from sklearn.metrics import mean_squared_error

import numpy as np
import matplotlib.pylab as pylab

# Version 1.0 - measured by MSE in one step
# Version 1.1 - measured by performance of our system in 3 steps

class interp_kriging:
    def __init__(self):
        return
    
    def interp(self, x, y, data, x_max, y_max, method):
        okri = pykrige.ok.OrdinaryKriging(x, y, data, variogram_model = method, nlags = 50)
        interp_arr2d, _ = okri.execute("grid", np.arange(x_max), np.arange(y_max))
        return interp_arr2d

class interp_opt:
    def __init__(self, proportion=0.2, algo_list=None):
        self.proportion = proportion
        self.kriging_method = ["linear", "gaussian", "exponential", "hole-effect"]
        self.interp2d_method = ['linear', 'cubic', 'quintic']
        if algo_list is None:
            self.algo_list = [interp_kriging, interp2d]
        else:
            self.algo_list = algo_list
            
    def data_split(self, x, y, data):
        assert len(x) == len(y) and len(x) == len(data)
        num = len(x)
        test = np.random.choice(num, int(num*self.proportion))
        test_mask = np.zeros_like(x).astype(bool)
        test_mask[test] = True
        test_set = (x[test_mask], y[test_mask], data[test_mask])
        train_set = (x[~test_mask], y[~test_mask], data[~test_mask])
        return train_set, test_set
    
    def optimize(self, x, y, data, x_max, y_max):
        train_set, test_set = self.data_split(x, y, data)
        train_x, train_y, train_data = train_set
        error_list = []
        for interp_algo in self.algo_list:
            if interp_algo == interp_kriging:
                
                for method in self.kriging_method:
                    interp_method = interp_algo()
                    result = interp_method.interp(train_x, train_y, train_data, float(x_max), float(y_max), method).T
                    error_list.append(mean_squared_error(result, feature_data))
            else:
                for method in self.interp2d_method:
                    interp_method = interp_algo(train_x, train_y, train_data, kind=method)
                    result = feature_data.copy()
                    for x in range(x_max):
                        for y in range(y_max):
                            if result[x, y] == 0:
                                result[x, y] = interp_method(x, y)
                    
                    error = mean_squared_error(result, feature_data)
                    error_list.append(error)
                    
                
        return error_list

if __name__ == '__main__':
    import rasterio
    import geopandas

    data_dir = 'Bayesian_main/dataset/Washington'
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
    # Data augment
    label_arr = label_arr2d[mask]
    
    geochemistry = geopandas.read_file(data_dir+'/shapefile/Geochemistry.shp')   
    feature_list = ['B', 'Ca', 'Cu', 'Fe', 'Mg', 'Ni']
    
    feature = 'Mg'
    feature_dict = {}
    size = mask_ds.index(mask_ds.bounds.right, mask_ds.bounds.bottom)
    global feature_data
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
    z = geochemistry[feature].values
    
    interpOPT = interp_opt()
    result = interpOPT.optimize(x_geo, y_geo, z, x_max, y_max)
    print(result)
    