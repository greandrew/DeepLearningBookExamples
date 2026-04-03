import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm.notebook import trange, tqdm

def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8):
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no*train_rate)]
    test_idx = idx[int(no*train_rate):]
    
    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]      
    
    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no*train_rate)]
    test_idx = idx[int(no*train_rate):]
  
    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]
  
    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time (data):
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:,0]))
        time.append(len(data[i][:,0]))
    
    return time, max_seq_len

class TimeSeriesDataset(Dataset):
    def __init__(self, data, time):
        self.data = data
        self.time = time

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.time[index]

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def discriminative_score_metrics(original_data, synthetic_data, device):
    original_data = original_data.squeeze()
    synthetic_data = synthetic_data.squeeze()
    
    # Split the data into training and test datasets
    train_original, test_original = train_test_split(original_data, test_size=0.2)
    train_synthetic, test_synthetic = train_test_split(synthetic_data, test_size=0.2)

    # Concatenate the training data and create a new array with flags for original and synthetic data
    train_data = np.concatenate((train_original, train_synthetic), axis=0)
    train_labels = np.concatenate((np.ones(len(train_original)), np.zeros(len(train_synthetic))))

    test_data = np.concatenate((test_original, test_synthetic), axis=0)
    test_labels = np.concatenate((np.ones(len(test_original)), np.zeros(len(test_synthetic))))

    # Convert to PyTorch tensors
    train_data, train_labels = torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32)
    test_data, test_labels = torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.float32)

    # Create DataLoader
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define Discriminator model, loss function, and optimizer
    input_dim = original_data.shape[1]
    model = Discriminator(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the Discriminator model
    num_epochs = 100
    tqdm_epoch = trange(num_epochs)
    for epoch in tqdm_epoch:
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data).squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    # Evaluate the model on the test data and calculate the classification error
    with torch.no_grad():
        test_data, test_labels = test_data.to(device), test_labels.to(device)
        test_outputs = model(test_data).squeeze()
        test_predictions = (test_outputs > 0.5).float()
        classification_error = (test_predictions != test_labels).float().mean().item()

    return classification_error

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Predictor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        outputs, _ = self.gru(x)
        y_hat = torch.sigmoid(self.fc(outputs[:, -1, :]))
        return y_hat

def predictive_score_metrics(original_data, synthetic_data, device, batch_size=128):
    
    original_data = np.stack(original_data)
    synthetic_data = np.stack(synthetic_data)
    # Basic Parameters

    no, seq_len, _ = np.asarray(original_data).shape

    # Convert data to tensors and unsqueeze the second dimension
    original_data = torch.tensor(original_data, dtype=torch.float32).to(device)
    synthetic_data = torch.tensor(synthetic_data, dtype=torch.float32).to(device)

    # Initialize the predictor
    hidden_dim = 6
    input_dim = seq_len - 1
    predictor = Predictor(input_dim, hidden_dim).to(device)

    # Loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(predictor.parameters())

    # Train the predictor on synthetic data using mini-batches
    iterations = 5000
    for itt in tqdm(range(iterations), desc='Training predictor'):
        # Create a mini-batch of synthetic data
        indices = np.random.choice(range(len(synthetic_data)), size=batch_size, replace=False)
        batch_synthetic_data = synthetic_data[indices]

        optimizer.zero_grad()
        y_pred = predictor(batch_synthetic_data[:, :-1, :].permute(0, 2, 1))
        loss = criterion(y_pred.squeeze(), batch_synthetic_data[:, -1, :].squeeze())
        loss.backward()
        optimizer.step()

    # Evaluate the predictor on original data
    with torch.no_grad():
        pred_Y_curr = predictor(original_data[:, :-1, :].permute(0,2,1))

    # Calculate the mean absolute error (MAE)
    predictive_score = mean_absolute_error(original_data[:, -1, :].squeeze().cpu().numpy(), pred_Y_curr.squeeze().cpu().numpy())

    return predictive_score

import pandas as pd

def calculate_scores(ori_data_list, gen_data_list, names_list, device):
    if len(ori_data_list) != len(gen_data_list) or len(ori_data_list) != len(names_list):
        raise ValueError("All input lists must have the same length")

    discriminative_scores = []
    predictive_scores = []

    for ori_data, gen_data, name in zip(ori_data_list, gen_data_list, names_list):
        dis_score = discriminative_score_metrics(ori_data, gen_data, device)
        print(f"Finished calculating discriminative score for {name}")
        pred_score = predictive_score_metrics(ori_data, gen_data, device)
        print(f"Finished calculating predictive score for {name}")
        discriminative_scores.append(dis_score)
        predictive_scores.append(pred_score)

    scores_df = pd.DataFrame({'Discriminative Scores': discriminative_scores,
                              'Predictive Scores': predictive_scores},
                             index=names_list)

    return scores_df


def real_data_loading(data: np.array, seq_len):
    """Load and preprocess real-world datasets.
    Args:
      - data_name: Numpy array with the values from a a Dataset
      - seq_len: sequence length
    Returns:
      - data: preprocessed data.
    """
    # Flip the data to make chronological data
    ori_data = data[::-1]

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        if(_x.ndim == 1):
            _x = _x[:, np.newaxis]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    return np.stack(data)