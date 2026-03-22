import torch
import matplotlib.pyplot as plt
import numpy as np

def scatter_plot(result, target, filename):
    plt.figure(figsize=(10, 10))
    
    # Scatter plot in grayscale
    plt.scatter(target.cpu().numpy(), result.cpu().numpy(), color='gray', alpha=0.5, edgecolors="lightgray", linewidth=0.5)
    
    plt.xlabel('Target Values', color='black')
    plt.ylabel('Model Outputs', color='black')
    
    # Grid in light gray
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray')
    
    # Axis colors in black
    plt.gca().tick_params(axis='both', colors='black')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, facecolor='white')
    plt.show()

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