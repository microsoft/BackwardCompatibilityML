import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):

    def __init__(self, input_size, num_classes, hidden_sizes=[50, 10]):
        super(MLPClassifier, self).__init__()
        layer_sizes = [input_size] + hidden_sizes + [num_classes]
        self.layers = [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]

        for i, layer in enumerate(self.layers):
            self.add_module("layer-%d" % i, layer)

    def forward(self, data, sample_weight=None):
        x = data
        out = x
        num_layers = len(self.layers)

        for i in range(num_layers):
            out = self.layers[i](out)
            if i < num_layers - 1:
                out = F.relu(out)

        out_softmax = F.softmax(out, dim=-1)
        out_log_softmax = F.log_softmax(out, dim=-1)

        return out, out_softmax, out_log_softmax


class LogisticRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        out_softmax = F.softmax(out, dim=-1)
        out_log_softmax = F.log_softmax(out, dim=-1)

        return out, out_softmax, out_log_softmax

