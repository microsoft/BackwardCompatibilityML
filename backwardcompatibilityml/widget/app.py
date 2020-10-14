# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from backwardcompatibilityml import loss as bcloss
from backwardcompatibilityml.helpers import training
from backwardcompatibilityml.helpers.models import LogisticRegression, MLPClassifier
from rai_core_flask.flask_helper import FlaskHelper
from .compatibility_analysis import CompatibilityAnalysis

folder_name = "tests/sweeps"
number_of_epochs = 10
batch_size_train = 70
batch_size_test = 139
learning_rate = 0.01
momentum = 0.5

dataset_file = open("tests/datasets/breast-cancer-wisconsin.data", "r")
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

h1 = LogisticRegression(9, 2)
optimizer = optim.SGD(h1.parameters(), lr=learning_rate, momentum=momentum)
train_counter, test_counter, train_losses, test_losses = training.train(
    number_of_epochs, h1, optimizer, F.nll_loss, partial_training_set, testing_set,
    batch_size_train, batch_size_test)
h1.eval()

with torch.no_grad():
    _, _, output = h1(testing_set[0][0])

h2 = MLPClassifier(9, 2)

analysis = CompatibilityAnalysis(
    folder_name,
    number_of_epochs,
    h1,
    h2,
    training_set,
    testing_set,
    batch_size_train,
    batch_size_test,
    lambda_c_stepsize=0.05,
    OptimizerClass=optim.SGD,
    optimizer_kwargs={"lr": learning_rate, "momentum": momentum},
    NewErrorLossClass=bcloss.BCCrossEntropyLoss,
    StrictImitationLossClass=bcloss.StrictImitationCrossEntropyLoss,
    port=5050)

app = FlaskHelper.app
app.logger.info('initialization complete')
