import os
import matplotlib.pyplot as plt

def create_file_path(path, name):
    # Create the full file path
    file_path = os.path.join(path, name)

    # Check if the directory exists and create it if it doesn't
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    return file_path

def plot_losses(loss_lists, loss_names, num_epochs, filename, dpi=300):
    if len(loss_lists) != len(loss_names):
        raise ValueError("The number of loss lists and loss names must be equal.")

    line_styles = ['-', '--', '-.', ':']

    plt.figure()
    for i, (losses, name) in enumerate(zip(loss_lists, loss_names)):
        plt.plot(range(1, num_epochs+1), losses, linestyle=line_styles[i % len(line_styles)], color='black', label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(filename, dpi=dpi)
    plt.show()
