import yfinance as yf
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset 
import random

def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data
    
def log_returns(time_series):
    time_series = np.asarray(time_series)
    
    if np.any(time_series <= 0):
        raise ValueError("Prices must be positive.")
    
    return np.diff(np.log(time_series), axis=0)

def normalize(df):
    mean = df.mean()
    std = df.std()
    normalized_df = (df - mean) / std
    return normalized_df, mean, std

def denormalize(normalized_df, mean, std):
    denormalized_df = normalized_df * std + mean
    return denormalized_df

class TimeSeriesDataset(Dataset):
    def __init__(self, time_series, window_size, condition_size):
        self.time_series = time_series
        self.window_size = window_size
        self.condition_size = condition_size

    def __len__(self):
        return len(self.time_series) - self.window_size - self.condition_size + 1

    def __getitem__(self, idx):
        start_idx = random.randint(0, len(self.time_series) - 
                                   self.window_size - self.condition_size)
        condition = self.time_series[start_idx : start_idx + self.condition_size]
        window = self.time_series[start_idx + self.condition_size : start_idx 
                                  + self.window_size + self.condition_size]
        return torch.tensor(condition, dtype=torch.float32), \
                torch.tensor(window, dtype=torch.float32)