# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

import os
import copy
import io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import random
from backwardcompatibilityml import loss as bcloss
from backwardcompatibilityml.helpers import training
from backwardcompatibilityml.helpers.models import LogisticRegression, MLPClassifier
from backwardcompatibilityml.widgets import ModelComparison
from flask import send_file
from PIL import Image
from rai_core_flask.flask_helper import FlaskHelper

use_ml_flow = True
ml_flow_run_name = "dev_app_sweep"


def breast_cancer_sweep():
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
        training_ids = list(range(prev, i+1))
        prev = i
        training_set_torch.append([training_ids, torch.stack(training_data, dim=0), torch.stack(training_labels, dim=0)])

    testing_set_torch = []
    prev = 0
    for i in range((batch_size_test - 1), len(testing_set), batch_size_test):
        testing_data = list(map(lambda r: r[0], testing_set[prev:(i + 1)]))
        testing_labels = list(map(lambda r: r[1], testing_set[prev:(i + 1)]))
        testing_ids = list(range(prev, i+1))
        prev = i
        testing_set_torch.append([testing_ids, torch.stack(testing_data, dim=0), torch.stack(testing_labels, dim=0)])

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
        _, _, output = h1(testing_set[0][1])

    h2 = MLPClassifier(9, 2)

    model_comparison = ModelComparison("sweeps-fico", number_of_epochs, h1, h2, training_set,
                                       use_ml_flow=True, device="cuda")

    # CompatibilityAnalysis(
    #     folder_name,
    #     number_of_epochs,
    #     h1,
    #     h2,
    #     training_set,
    #     testing_set,
    #     batch_size_train,
    #     batch_size_test,
    #     lambda_c_stepsize=0.05,
    #     OptimizerClass=optim.SGD,
    #     optimizer_kwargs={"lr": learning_rate, "momentum": momentum},
    #     NewErrorLossClass=bcloss.BCCrossEntropyLoss,
    #     StrictImitationLossClass=bcloss.StrictImitationCrossEntropyLoss,
    #     use_ml_flow=use_ml_flow,
    #     ml_flow_run_name=ml_flow_run_name)


def mnist_sweep():
    sweeps_folder = "development/model-comparison/sweeps-mnist"
    n_epochs = 3
    n_samples = 5000
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    download = not os.path.exists("examples/datasets/MNIST")
    data_set = torchvision.datasets.MNIST('examples/datasets/', train=True, download=download,
                                          transform=transform)
    data_loader = list(torch.utils.data.DataLoader(data_set, shuffle=True))

    instance_ids = list(range(len(data_loader)))
    dataset = []
    for (instance_id, (data_instance, data_label)) in zip(instance_ids, data_loader):
        dataset.append([instance_id, data_instance.view(1, 28, 28), data_label.view(1)])

    training_set = dataset[:int(0.8*n_samples)]
    testing_set = dataset[int(0.8*n_samples):]

    train_loader = []
    prev = 0
    for i in range((batch_size_train - 1), len(training_set), batch_size_train):
        batch_ids = list(map(lambda r: r[0], training_set[prev:i]))
        training_data = list(map(lambda r: r[1], training_set[prev:i]))
        training_labels = list(map(lambda r: r[2], training_set[prev:i]))
        prev = i
        train_loader.append([batch_ids, torch.stack(training_data, dim=0), torch.stack(training_labels, dim=0).view(len(training_labels))])

    test_loader = []
    prev = 0
    for i in range((batch_size_test - 1), len(testing_set), batch_size_test):
        batch_ids = list(map(lambda r: r[0], testing_set[prev:i]))
        testing_data = list(map(lambda r: r[1], testing_set[prev:i]))
        testing_labels = list(map(lambda r: r[2], testing_set[prev:i]))
        prev = i
        test_loader.append([batch_ids, torch.stack(testing_data, dim=0), torch.stack(testing_labels, dim=0).view(len(testing_labels))])

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return x, F.softmax(x, dim=1), F.log_softmax(x, dim=1)

    h1 = Net().cuda()
    state_file = f"{sweeps_folder}/h1-model.state"
    if (os.path.exists(state_file)):
        h1.load_state_dict(torch.load(state_file))
        print("Loaded state dict")
    else:
        optimizer = optim.SGD(h1.parameters(), lr=learning_rate, momentum=momentum)
        print("Training model")
        train_counter, test_counter, train_losses, test_losses = training.train(
            n_epochs, h1, optimizer, F.nll_loss, train_loader, test_loader,
            batch_size_train, batch_size_test, device="cuda")
        torch.save(h1.state_dict(), state_file)
        print("Saved state dict")

    h2 = Net().cuda()

    def unnormalize(img):
        img = img / 2 + 0.5
        return img

    def get_instance_image(instance_id):
        img_bytes = io.BytesIO()
        data = np.reshape(
            np.uint8(np.transpose((unnormalize(dataset[instance_id][1])), (1, 2, 0)).numpy() * 255),
            (28, 28))
        img = Image.fromarray(data)
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        return send_file(img_bytes, mimetype='image/png')

    def get_instance_label(instance_id):
        label = dataset[instance_id][2].item()
        return f'{label}'

    model_comparison = ModelComparison("sweeps-fico", n_epochs, h1, h2, train_loader,
                                       get_instance_image_by_id=get_instance_image,
                                       use_ml_flow=True, device="cuda")

    # CompatibilityAnalysis(sweeps_folder, n_epochs, h1, h2, train_loader, test_loader,
    #                       batch_size_train, batch_size_test,
    #                       OptimizerClass=optim.SGD,
    #                       optimizer_kwargs={"lr": learning_rate, "momentum": momentum},
    #                       NewErrorLossClass=bcloss.BCCrossEntropyLoss,
    #                       StrictImitationLossClass=bcloss.StrictImitationCrossEntropyLoss,
    #                       lambda_c_stepsize=0.25,
    #                       get_instance_image_by_id=get_instance_image,
    #                       get_instance_metadata=get_instance_label,
    #                       device="cuda",
    #                       use_ml_flow=use_ml_flow,
    #                       ml_flow_run_name=ml_flow_run_name)


mnist_sweep()
app = FlaskHelper.app
app.logger.info('initialization complete')
