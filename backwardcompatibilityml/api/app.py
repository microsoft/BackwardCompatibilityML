from flask import Flask
from flask import Response
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
import random
from backwardcompatibilityml import loss as bcloss
from backwardcompatibilityml.helpers import training
from backwardcompatibilityml.sweep_management import SweepManager

app = Flask(__name__)

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

def create_sweep_manager():
    print("Creating sweep manager")
    print("Reading dataset")
    dataset_file = open("../../tests/datasets/breast-cancer-wisconsin.data", "r")
    raw_data = dataset_file.read()
    dataset_file.close()
    data = list(map(lambda l: l.split(",")[1:], filter(lambda l: len(l) > 0, map(lambda l: l.strip(), raw_data.split("\n")))))
    for i in range(len(data)):
        row = data[i]
        row = list(map(lambda e: 0 if (e == "?") else int(e), row))
        if row[9] == 2:
            row[9] = 0
        elif row[9] == 4:
            row[9] = 1
        data[i] = row

    data = list(map(lambda r: (torch.tensor(r[:9], dtype=torch.float32), torch.tensor(r[9])), data))

    random.shuffle(data)

    folder_name = "sweeps"
    number_of_epochs = 10
    batch_size_train = 70
    batch_size_test = 139
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    training_set = data[:560]
    testing_set = data[560:]

    training_set_torch = []
    prev = 0
    for i in range((batch_size_train - 1), len(training_set), batch_size_train):
        training_data = list(map(lambda r: r[0], training_set[prev:(i + 1)]))
        training_labels = list(map(lambda r: r[1], training_set[prev:(i + 1)]))
        prev = i
        training_set_torch.append([torch.stack(training_data, dim=0), torch.stack(training_labels, dim=0)])

    testing_set_torch = []
    prev = 0
    for i in range((batch_size_test - 1), len(testing_set), batch_size_test):
        testing_data = list(map(lambda r: r[0], testing_set[prev:(i + 1)]))
        testing_labels = list(map(lambda r: r[1], testing_set[prev:(i + 1)]))
        prev = i
        testing_set_torch.append([torch.stack(testing_data, dim=0), torch.stack(testing_labels, dim=0)])

    training_set = training_set_torch
    partial_training_set = training_set[:int(0.5 * (len(training_set)))]
    testing_set = testing_set_torch

    print("Training h1")
    h1 = LogisticRegression(9, 2)
    optimizer = optim.SGD(h1.parameters(), lr=learning_rate, momentum=momentum)
    train_counter, test_counter, train_losses, test_losses = training.train(
        number_of_epochs, h1, optimizer, F.nll_loss, partial_training_set, testing_set,
        batch_size_train, batch_size_test)
    h1.eval()

    with torch.no_grad():
        _, _, output = h1(testing_set[0][0])

    h1_accuracy = accuracy_score(output.data.max(1)[1].numpy(), testing_set[0][1].numpy())

    print("Returning SweepManager")
    h2 = MLPClassifier(9, 2)
    return SweepManager(
        folder_name,
        number_of_epochs,
        h1,
        h2,
        training_set,
        testing_set,
        batch_size_train,
        batch_size_test,
        optim.SGD,
        {"lr": learning_rate, "momentum": momentum},
        bcloss.BCCrossEntropyLoss,
        bcloss.StrictImitationCrossEntropyLoss,
        lambda_c_stepsize=0.05)

sweep_manager = create_sweep_manager()

@app.route("/api/v1/start_sweep", methods=["POST"])
def start_sweep():
    sweep_manager.start_sweep()
    return {
        "running": sweep_manager.sweep_thread.is_alive(),
        "percent_complete": sweep_manager.get_sweep_status()
    }

@app.route("/api/v1/sweep_status", methods=["GET"])
def get_sweep_status():
    return {
        "running": sweep_manager.sweep_thread.is_alive(),
        "percent_complete": sweep_manager.get_sweep_status()
    }

@app.route("/api/v1/sweep_summary", methods=["GET"])
def get_data():
    return Response(
        json.dumps(sweep_manager.get_sweep_summary()),
        mimetype="application/json")

@app.route("/api/v1/evaluation_data/<int:evaluation_id>")
def get_evaluation(evaluation_id):
    return sweep_manager.get_evaluation(evaluation_id)

if __name__ == '__main__':
    app.run(debug=True)