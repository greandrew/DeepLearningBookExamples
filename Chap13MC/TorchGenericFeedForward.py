import torch.nn as nn

class NeuralNetVariable(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden, output_size):
        super(NeuralNetVariable, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        for i in range(num_hidden):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        out = x
        for ilayer in self.layers:
            out = ilayer(out)
        return out

class NeuralNetVariableLeaky(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden, output_size):
        super(NeuralNetVariableLeaky, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.LeakyReLU())
        for i in range(num_hidden):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        out = x
        for ilayer in self.layers:
            out = ilayer(out)
        return out

class NeuralNetVariableDropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden, output_size, prob):
        super(NeuralNetVariableDropout, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=prob))
        for i in range(num_hidden):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=prob))
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        out = x
        for ilayer in self.layers:
            out = ilayer(out)
        return out