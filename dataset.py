#%%
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

# %%
from torch.utils.data import DataLoader
def load_traffic(root, batch_size, n_workers, mode):
    """
    Load traffic dataset
    return train_loader, val_loader, test_loader
    """
    df = pd.read_hdf(root)
    df = df.reset_index()
    df = df.rename(columns={"index":"utc"})
    df["utc"] = pd.to_datetime(df["utc"], unit="s")
    df = df.set_index("utc")
    n_sensor = len(df.columns)

    mean = df.values.flatten().mean()
    std = df.values.flatten().std()

    df = (df - mean)/std
    df = df.sort_index()
    # split the dataset
    train_df = df.iloc[:int(0.75*len(df))]
    val_df = df.iloc[int(0.75*len(df)):int(0.875*len(df))]
    test_df = df.iloc[int(0.75*len(df)):]

    if(mode=="debug"):
        print("DEBUG MODE: LOADING DATASET WITH 10% DATA ")
        train_df = train_df.sample(n=int(0.1*len(train_df)))
        val_df = val_df.sample(n=int(0.1*len(train_df)))
        test_df = test_df.sample(n=int(0.1*len(train_df)))


    train_loader = DataLoader(Traffic(train_df), batch_size=batch_size, shuffle=True, num_workers=n_workers, persistent_workers=(n_workers!=0))
    val_loader = DataLoader(Traffic(val_df), batch_size=batch_size, shuffle=False, num_workers=n_workers, persistent_workers=(n_workers!=0))
    test_loader = DataLoader(Traffic(test_df), batch_size=batch_size, shuffle=False, num_workers=n_workers, persistent_workers=(n_workers!=0))

    return train_loader, val_loader, test_loader, n_sensor  

class Traffic(Dataset):
    def __init__(self, df, window_size=12, stride_size=1):
        super(Traffic, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.time = self.preprocess(df)
    
    def preprocess(self, df):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        delat_time =  df.index[end_idx]-df.index[start_idx]
        idx_mask = delat_time==pd.Timedelta(5*self.window_size,unit='min')

        return df.values, start_idx[idx_mask], df.index[start_idx[idx_mask]]

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])

        return torch.FloatTensor(data).transpose(0,1)

def load_water(root, batch_size, lookback, label=False):
    
    data = pd.read_csv(root)
    data = data.rename(columns={"Normal/Attack":"label"})
    data.label[data.label!="Normal"]=1
    data.label[data.label=="Normal"]=0
    data["Timestamp "] = pd.to_datetime(data["Timestamp "])
    data = data.set_index("Timestamp ")

    #%%
    feature = data.iloc[:,:51]
    mean_df = feature.mean(axis=0)
    std_df = feature.std(axis=0)

    norm_feature = (feature-mean_df)/std_df
    norm_feature = norm_feature.dropna(axis=1)
    n_sensor = len(norm_feature.columns)

    train_df = norm_feature.iloc[:int(0.6*len(data))]
    train_label = data.label.iloc[:int(0.6*len(data))]

    val_df = norm_feature.iloc[int(0.6*len(data)):int(0.7*len(data))]
    val_label = data.label.iloc[int(0.6*len(data)):int(0.7*len(data))]
    
    test_df = norm_feature.iloc[int(0.7*len(data)):]
    test_label = data.label.iloc[int(0.7*len(data)):]
    if label:
        train_loader = DataLoader(WaterLabel(train_df,train_label, lookback), batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(Water(train_df,train_label, lookback), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Water(val_df,val_label, lookback), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Water(test_df,test_label, lookback), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor

class Water(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10):
        super(Water, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df,label)
    
    def preprocess(self, df, label):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        delat_time =  df.index[end_idx]-df.index[start_idx]
        idx_mask = delat_time==pd.Timedelta(self.window_size,unit='s')

        return df.values, start_idx[idx_mask], label[start_idx[idx_mask]]

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])

        return torch.FloatTensor(data).transpose(0,1)


class WaterLabel(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10):
        super(WaterLabel, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df,label)
        self.label = 1.0-2*self.label 
    
    def preprocess(self, df, label):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        delat_time =  df.index[end_idx]-df.index[start_idx]
        idx_mask = delat_time==pd.Timedelta(self.window_size,unit='s')

        return df.values, start_idx[idx_mask], label[start_idx[idx_mask]]

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])

        return torch.FloatTensor(data).transpose(0,1),self.label[index]
    
def load_act(root, batch_size, lookback, cut=1, resample_rate=1, label=False, train_test_split=0.7, scaler='Standard', spectral_residual=False ):
    from os import listdir, makedirs, path
    X_1 = pd.read_csv(path.join(root, 'Train/X_train.txt'), delimiter=' ', header=None)
    X_2 = pd.read_csv(path.join(root, 'Test/X_test.txt'), delimiter=' ', header=None)

    values = pd.concat([X_1, X_2], axis=0, ignore_index=True)

    y_1 = pd.read_csv(path.join(root, 'Train/y_train.txt'), delimiter=' ', header=None)
    y_2 = pd.read_csv(path.join(root, 'Test/y_test.txt'), delimiter=' ', header=None)
    y = pd.concat([y_1, y_2], axis=0, ignore_index=True)
    labels = np.array([x in range(7,13) for x in y.values])

    if cut < 1:
        print('Cutting the dataset at ' + str(cut) + ' length \n')
        values = values.iloc[:int(len(values)*cut)]
        labels = labels[:int(len(labels)*cut)]
    sample_rate = resample_rate
    if sample_rate<=0 or sample_rate>1:
        print('Incorrect resample rate, defaulting to 1\n')
        sample_rate = 1
    else:
        print('resampling to one observation every '+ str(int(1/sample_rate)))

    values = values.iloc[::int(1/sample_rate)]#resampling
    labels = pd.Series(labels[::int(1/sample_rate)])#resampling

    train_test_split=train_test_split

    if scaler == 'quantile':
        from sklearn.preprocessing  import QuantileTransformer
        scaler = QuantileTransformer(output_distribution='uniform')
    if scaler =='standard':
        from sklearn.preprocessing  import StandardScaler
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing  import MinMaxScaler
        scaler = MinMaxScaler()

    values = pd.DataFrame(scaler.fit_transform(values)) 
    
    train_values = values.iloc[:int((train_test_split-0.1)*len(labels)),:]
    val_values = values.iloc[int((train_test_split-0.1)*len(labels)):int((train_test_split)*len(labels)),:].reset_index(drop=True)
    
    train_labels = labels[:int((train_test_split-0.1)*len(labels))]
    val_labels = labels[int((train_test_split-0.1)*len(labels)):int((train_test_split)*len(labels))].reset_index(drop=True)

    print('removing anomalies from training data')
    train_values = train_values[train_labels==False].reset_index(drop=True)
    train_labels = train_labels[train_labels==False].reset_index(drop=True)

    test_values = values.iloc[int(train_test_split*len(labels)):,:].reset_index(drop=True)
    test_labels = labels[int(train_test_split*len(labels)):].reset_index(drop=True)

    train_loader = DataLoader(Act(train_values,train_labels, lookback), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Act(val_values,val_labels, lookback), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Act(test_values,test_labels, lookback), batch_size=batch_size, shuffle=False)

    n_sensor = values.shape[1]
    return train_loader, val_loader, test_loader, n_sensor

class Act(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10):
        super(Act, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df,label)
    
    def preprocess(self, df, label):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        delat_time =  df.index[end_idx]-df.index[start_idx]
        idx_mask = delat_time==pd.Timedelta(self.window_size,unit='s')

        return df.values, start_idx, label[start_idx]

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])

        return torch.FloatTensor(data).transpose(0,1)
