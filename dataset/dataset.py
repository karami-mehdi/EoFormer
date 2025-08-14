import os
import json
import numpy as np
from PIL import Image
import torch 
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from monai.data import CacheDataset
from monai.transforms import Resize, MapTransform
from google.colab import drive

def get_dataset_brats(data_path: str, 
                  json_file: str, 
                  transform=None,
                 )-> Tuple[Dataset, Dataset]:
    
    drive.mount('/content/drive')
    with np.load('/content/drive/My Drive/processedDatasetFolds/fold_1.npz') as data:
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

    rng = np.random.RandomState(42)
    n = len(X_train)
    idx = rng.permutation(n)
    split = int(n * 0.8)
    tr_idx = idx[:split]
    va_idx = idx[split:]

    X_tr = X_train[tr_idx]
    y_tr = y_train[tr_idx]
    X_va = X_train[va_idx]
    y_va = y_train[va_idx]

    train_list = [{"image": X_tr[i], "label": y_tr[i]} for i in range(len(X_tr))]
    valid_list = [{"image": X_va[i], "label": y_va[i]} for i in range(len(X_va))]
    test_list  = [{"image": X_test[i], "label": y_test[i]} for i in range(len(X_test))]

    train_set = CacheDataset(
                            data = train_list,
                            cache_rate=0.0,
                            transform = transform['train']
                            )
    valid_set = CacheDataset(
                            data = valid_list,
                            cache_rate=0.0,
                            transform = transform['valid']
                            )
    test_set = CacheDataset(
                            data = test_list,
                            cache_rate=0.0,
                            transform = transform['valid']
                            )
    
    return train_set, valid_set, test_set, train_list, valid_list, test_list