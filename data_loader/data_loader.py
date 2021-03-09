import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler

import torch
from torch.utils.data import TensorDataset, DataLoader

def load_data( data_dir, input_filename, time_name, out_feature, in_features, batch_size, window_size, num_workers, shuffle):

    kwargs = {"num_workers": num_workers, "pin_memory": True}
    
    # read dataset
    print( "Loading Data..." )
    file_path = os.path.join( data_dir, input_filename )
    df = pd.read_csv(filepath_or_buffer = file_path, low_memory = False)

    # remove timestamp
    if time_name in df.columns:
        df = df.drop( time_name, axis = 1 )

    # data & label
    if out_feature != []:
        df_label = df[ out_feature ]
        df_data  = df[ in_features ]
    else:
        df_label = df
    
# =============================================================================
#     # remove ',' and transform string to float
#     for i in list(df_data): 
#         df_data[i] = df_data[i].apply(lambda x: str(x).replace("," , ""))
#     df_data = df_data.astype(float)
# =============================================================================
    
    # NaN or empty check
    assert np.all( ~pd.isna(df_data).values )
    
    # Calculate normalization
    df_data_orig = df_data.values
    scaler = RobustScaler( quantile_range = (1, 99)).fit( df_data_orig )
    
    # Adopt normalization
    df_data_orig = scaler.transform( df_data_orig )
    df_data_orig = pd.DataFrame( df_data_orig )
    
    # Group multiple data points into a chunk (aka window_size)
    windows_df = df_data_orig.values[np.arange(window_size)[None, :] + np.arange(df_data_orig.shape[0]-window_size)[:, None]]
    
    # define input feature size
    sample_size = windows_df.shape[0]
    input_size = windows_df.shape[1] * windows_df.shape[2]
    
    # for data, transform N x H x W to N x (H x W) format
    data = torch.from_numpy( windows_df ).view( sample_size, input_size )
    
    # generate dataset and loader
    data = data.float()
    data = TensorDataset( data )
    data_loader = DataLoader( dataset = data, batch_size = batch_size, shuffle = shuffle, **kwargs )
    
    return data_loader, scaler, sample_size, input_size

