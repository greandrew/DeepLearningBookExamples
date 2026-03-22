import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm.notebook import tqdm, trange

def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs):
    train_errors = []
    test_errors = []

    tqdm_epoch = trange(epochs)
    for epoch in tqdm_epoch:
        model.train()
        train_loss = 0.0

        # Training
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = loss_fn(outputs.squeeze(), batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)
        train_errors.append(train_loss)
        
        # Evaluation on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = F.mse_loss(outputs.squeeze(), batch_y)  # Assuming MSE is the desired error metric
                test_loss += loss.item() * batch_X.size(0)

        test_loss /= len(test_loader.dataset)
        test_errors.append(test_loss)

        tqdm_epoch.set_description(f"Epoch {epoch+1}/{epochs} - Train error: {train_loss:.4f}, Test error: {test_loss:.4f}")

    return train_errors, test_errors

# Swish activation function
class Swish(nn.Module):
    def forward(self, input):
        return input * torch.sigmoid(input)
    
    # Feed-forward network
class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, num_hidden_layers, neurons_per_layer):
        super(FeedForwardNetwork, self).__init__()

        # Construct layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, neurons_per_layer))
        layers.append(Swish())

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(Swish())

        # Output layer
        layers.append(nn.Linear(neurons_per_layer, 1))

        # Combine layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
import QuantLib as ql
import math

def black_scholes(option_type, strike, vol, T, spot, rate):
    option_type_mapping = {
        1: ql.Option.Call,
        0: ql.Option.Put,
        'call': ql.Option.Call,
        'put': ql.Option.Put
    }
    
    optype = option_type_mapping.get(option_type, None)
    
    if optype is None:
        raise ValueError("Invalid option_type. Use 'call' or 1 for Call option, or 'put' or 0 for Put option.")
        
    payoff = ql.PlainVanillaPayoff(optype, strike)
        
    stddev = vol * math.sqrt(T)
    discount = math.exp(-rate * T)
    
    return ql.BlackCalculator(payoff, spot, stddev, discount)