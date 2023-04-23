import os
import numpy as np
import pandas as pd
import pickle
import torch
import random
from datetime import datetime
from torch import nn


import torch
import torch.nn as nn

class Normalizer():
    def __init__(self, eps=1e-5):
        self.eps = eps

    def forward(self, x):
        self._get_statistics(x)
        x = self._normalize(x)
        return x
    
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        return x



# TODO: Recheck logic of this function
def create_3d_array(df, indices, T):
    n = len(indices)
    F = df.shape[1]
    df = df.sort_index()
    
    # result = np.zeros((n, T, F))
    result = []

    # create dictionary where key is index and value is next index
    next_idx_dict = {df.index[i]: df.index[i+1] for i in range(len(df)-1)}
    
    for i, idx in enumerate(indices):
        if idx not in next_idx_dict: 
            # print(f"Index {idx} is the last index")
            continue
        temp_df = df.loc[:next_idx_dict[idx]]
        if len(temp_df) >= T:
            # result[i] = temp_df.iloc[-T:].values
            result.append(temp_df.iloc[-T:].values)
        # else:
            # print(f"Index {idx} has less than {T} rows")


    return np.array( result )

def generate_univariate_data_labels(array):

    data, labels = [], []
    num_points   = array.shape[0]
    len_ts       = array.shape[1]
    num_channels = array.shape[2]

    for i in range(num_points - len_ts):
        for channel in range(num_channels):
            x = array[i][:, channel]
            y = array[i+len_ts][:, channel]

            x = np.expand_dims(x, axis=1)
            y = np.expand_dims(y, axis=1)

            data.append(x)
            labels.append(y)
        
    data     = np.array(data)
    labels   = np.array(labels)

    return data, labels
    
def generate_data_labels_from_3d_array(array):

    data, labels = [], []
    num_points   = array.shape[0]
    len_ts       = array.shape[1]
    num_channels = array.shape[2]

    for i in range(num_points - len_ts):
        
        x = array[i][:, ]
        y = array[i+len_ts][:, ]

        data.append(x)
        labels.append(y)
        
    data     = np.array(data)
    labels   = np.array(labels)

    return data, labels


def standardize(df, look_back, type='standard'):
    """
    Standardize a dataframe using rolling mean and std
    
    """
    if type == 'minmax':
        x_bar   = df.rolling(look_back).min()
        z_std   = df.rolling(look_back).max() - x_bar
        z_score = 1 + (df - x_bar) / z_std
    elif type=='standard':
        x_bar   = df.rolling(look_back).mean()
        z_std   = df.rolling(look_back).std()
        z_score = (df - x_bar) / z_std
    else:
        raise ValueError("type must be 'standard' or 'minmax'")

    z_score.replace([np.inf, -np.inf], np.nan, inplace=True)
    return z_score.fillna(method ='ffill').dropna()


def rolling_mean_diff(df, look_back_windows, type='standard'):
    """
    Calculate rolling mean difference

    Args:
        df (_type_): _description_
        look_back_windows (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = df.copy()
    result = []
    for look_back in look_back_windows:
        # temp_df = df - df.rolling(look_back).mean()
        if type == 'standard' or type == 'minmax':
            temp_df = standardize(df, look_back, type=type)
        elif type == 'max':
            temp_df = df / df.rolling(look_back).max()
        temp_df.columns = [f'{col}_{type}_{look_back}' for col in temp_df.columns]
        result.append(temp_df)
    result = pd.concat(result, axis=1).fillna(method='ffill').dropna()
    return result


# Cite this code as:
# Han, Yue-Zhi. (2021, January 1). ts2vec/utils.py [Computer software]. GitHub. https://github.com/yuezhihan/ts2vec/blob/main/utils.py#L47


# Cite this code as:
# Kim, Tae-Suk. (2021, January 1). ts2vec/REVIN.py [Computer software]. GitHub. https://github.com/ts-kim/RevIN/blob/master/RevIN.py


def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr
    
def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]

def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B*T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B*T,
        size=int(B*T*p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res

def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")



# https://stackoverflow.com/questions/44130851/simple-lstm-in-pytorch-with-sequential-module
class ExtractTensor(nn.Module):
    def forward(self,input, return_type='hidden'):
        # Output shape (batch, features, hidden)
        x, (hidden_n, cell_n) = input
        # Reshape shape (batch, hidden)
        if return_type == 'hidden':
            x = hidden_n[-1]
        elif return_type == 'cell':
            x = cell_n[-1]

        return x
    

def mask_input(x, mask_percentage=0.2):
    # The mask_percentage should be between 0 and 1
    assert 0 <= mask_percentage <= 1
    
    # Generate a mask tensor with the same shape as x
    mask = torch.rand_like(x) < mask_percentage
    masked_x = x.clone()  # Create a copy to not modify the original input
    masked_x[mask] = 0  # Set masked elements to 0 (or any desired value)
    return masked_x


def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]


