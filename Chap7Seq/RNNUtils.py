import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm.notebook import trange
import matplotlib.pyplot as plt

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
            loss = loss_fn(outputs.squeeze(), batch_y.squeeze())
            
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
                loss = loss_fn(outputs.squeeze(), batch_y.squeeze())  
                test_loss += loss.item() * batch_X.size(0)

        test_loss /= len(test_loader.dataset)
        test_errors.append(test_loss)

        tqdm_epoch.set_description(f"Epoch {epoch+1}/{epochs} - Train error: {train_loss:.4f}, Test error: {test_loss:.4f}")

    return train_errors, test_errors

def plot_errors(train_errors, test_errors, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(train_errors, color='gray', linestyle='-', label='Train Error')
    plt.plot(test_errors, color='black', linestyle='--', label='Test Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Training and Test Errors')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def train_classifier_model(model, train_loader, test_loader, loss_fn, optimizer, epochs):
    train_errors = []
    test_errors = []
    train_accuracies = []
    test_accuracies = []

    tqdm_epoch = trange(epochs)
    for epoch in tqdm_epoch:
        model.train()
        train_loss = 0.0
        correct_train = 0

        # Training
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = loss_fn(outputs.squeeze(), batch_y.float())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

            # Calculate training accuracy
            predicted = torch.sigmoid(outputs).squeeze() > 0.5
            correct_train += (predicted == batch_y).sum().item()

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100 * correct_train / len(train_loader.dataset)
        train_errors.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluation on test set
        model.eval()
        test_loss = 0.0
        correct_test = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = loss_fn(outputs.squeeze(), batch_y.float())
                test_loss += loss.item() * batch_X.size(0)

                # Calculate test accuracy
                predicted = torch.sigmoid(outputs).squeeze() > 0.5
                correct_test += (predicted == batch_y).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100 * correct_test / len(test_loader.dataset)
        test_errors.append(test_loss)
        test_accuracies.append(test_accuracy)

        tqdm_epoch.set_description(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

    return train_errors, test_errors, train_accuracies, test_accuracies

def plot_errors_accuracy(train_errors, test_errors, train_accuracies, test_accuracies, filename):
    plt.figure(figsize=(10, 6))

    # Plotting errors
    plt.plot(train_errors, color='black', linestyle='-', label='Train Error')
    plt.plot(test_errors, color='black', linestyle='--', label='Test Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend(loc='upper left')

    # Creating a second y-axis for accuracies
    ax2 = plt.twinx()
    ax2.plot(train_accuracies, color='black', linestyle=':', label='Train Accuracy', alpha=0.7)
    ax2.plot(test_accuracies, color='black', linestyle='-.', label='Test Accuracy', alpha=0.7)
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='upper right')

    # Title and grid
    plt.title('Training and Test Errors and Accuracies')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Layout and saving
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
