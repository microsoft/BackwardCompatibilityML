# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

import os
import json
import copy
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import random
from backwardcompatibilityml import loss as bcloss
from backwardcompatibilityml.helpers import training
from backwardcompatibilityml.helpers.models import LogisticRegression, MLPClassifier
from backwardcompatibilityml.metrics import model_accuracy


def get_sweep_data(folder_name, datapoint_index):
    sweep_file = open(f"{folder_name}/{datapoint_index}-evaluation-data.json", "r")
    sweep_data = json.loads(sweep_file.read())
    sweep_file.close()
    return sweep_data


def get_sweep_model_state_filename(folder_name, lambda_c, new_error=False, strict_imitation=False):
    if new_error and not strict_imitation:
        model_type = "new-error"
    elif not new_error and strict_imitation:
        model_type = "strict-imitation"

    state_filename = f"{folder_name}/{lambda_c}-model-{model_type}.state"

    return state_filename


def get_sweep_entry(sweep_summary, lambda_c, train=False, test=False, strict_imitation=False, new_error=False):
    result = list(
        filter(
            lambda entry: (
                entry["lambda_c"] - lambda_c < 1e-5 and entry["training"] == train and entry["testing"] == test and entry["strict-imitation"] == strict_imitation and entry["new-error"] == new_error),
            sweep_summary["data"]))
    if len(result) > 0:
        return result.pop()

    return None


class TestLossFunctions(object):

    @classmethod
    def setup_class(cls):
        """Set up once for all tests."""
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

        instance_ids = list(range(len(data)))
        cls.data_rows = list(map(lambda r: [r[0], r[1][0:9], r[1][9]], list(zip(instance_ids, data))))

        data = list(map(lambda r: (r[0], torch.tensor(r[1], dtype=torch.float32), torch.tensor(r[2])), cls.data_rows))

        random.shuffle(data)
        cls.n_epochs = 10
        cls.batch_size_train = 70
        cls.batch_size_test = 139
        cls.learning_rate = 0.01
        cls.momentum = 0.5
        cls.log_interval = 10
        random_seed = 1
        torch.manual_seed(random_seed)

        training_set = data[:560]
        testing_set = data[560:]

        training_set_torch = []
        prev = 0
        for i in range((cls.batch_size_train - 1), len(training_set), cls.batch_size_train):
            batch_ids = list(map(lambda r: r[0], training_set[prev:(i + 1)]))
            training_data = list(map(lambda r: r[1], training_set[prev:(i + 1)]))
            training_labels = list(map(lambda r: r[2], training_set[prev:(i + 1)]))
            prev = i
            training_set_torch.append([batch_ids, torch.stack(training_data, dim=0), torch.stack(training_labels, dim=0)])

        testing_set_torch = []
        prev = 0
        for i in range((cls.batch_size_test - 1), len(testing_set), cls.batch_size_test):
            batch_ids = list(map(lambda r: r[0], testing_set[prev:(i + 1)]))
            testing_data = list(map(lambda r: r[1], testing_set[prev:(i + 1)]))
            testing_labels = list(map(lambda r: r[2], testing_set[prev:(i + 1)]))
            prev = i
            testing_set_torch.append([batch_ids, torch.stack(testing_data, dim=0), torch.stack(testing_labels, dim=0)])

        cls.training_set = training_set_torch
        cls.partial_training_set = cls.training_set[:int(0.5 * (len(cls.training_set)))]
        cls.testing_set = testing_set_torch

    def setup(self):
        shutil.rmtree("tests/sweeps", ignore_errors=True)
        os.mkdir("tests/sweeps")

    def teardown(self):
        shutil.rmtree("tests/sweeps", ignore_errors=True)
        os.mkdir("tests/sweeps")

    def test_compatibility_sweep(self):
        h1 = LogisticRegression(9, 2)
        optimizer = optim.SGD(h1.parameters(), lr=self.learning_rate, momentum=self.momentum)
        train_counter, test_counter, train_losses, test_losses = training.train(
            self.n_epochs, h1, optimizer, F.nll_loss, self.partial_training_set, self.testing_set,
            self.batch_size_train, self.batch_size_test)

        h1.eval()

        with torch.no_grad():
            _, _, output = h1(self.testing_set[0][1])

        h1_accuracy = accuracy_score(output.data.max(1)[1].numpy(), self.testing_set[0][2].numpy())

        h2 = MLPClassifier(9, 2)

        training.compatibility_sweep(
            "tests/sweeps", self.n_epochs, h1, h2, self.training_set, self.testing_set,
            self.batch_size_train, self.batch_size_test,
            optim.SGD, {"lr": self.learning_rate, "momentum": self.momentum},
            bcloss.BCCrossEntropyLoss, bcloss.StrictImitationCrossEntropyLoss,
            lambda_c_stepsize=0.05)

        sweep_summary_file = open("tests/sweeps/sweep_summary.json", "r")
        sweep_summary = json.loads(sweep_summary_file.read())
        testing_data_new_error = list(filter(lambda row: row["new-error"] and row["testing"], sweep_summary["data"]))
        lambda_c_values = list(map(lambda row: row["lambda_c"], testing_data_new_error))
        assert(len(lambda_c_values) == len(np.arange(0.0, 1.0 + (0.05 / 2), 0.05)))

        entry_lambda_c_0 = list(filter(lambda row: row["lambda_c"] == 0.0, testing_data_new_error)).pop()
        entry_lambda_c_1 = list(filter(lambda row: row["lambda_c"] == 1.0, testing_data_new_error)).pop()

        # Todo: Need to find a better dataset on which the relative performance,
        # btc and bec betweeh h1 and h2, behave as expected
        assert(entry_lambda_c_0["performance"] >= h1_accuracy)
        # assert(entry_lambda_c_1["performance"] <= entry_lambda_c_0["performance"])
        # assert(entry_lambda_c_1["bec"] >= entry_lambda_c_0["bec"])
        lambda_step = 0.0
        while lambda_step <= 1.0:
            train_sweep_entry = get_sweep_entry(
                sweep_summary, lambda_step,
                train=True, test=False, strict_imitation=False, new_error=True)
            test_sweep_entry = get_sweep_entry(
                sweep_summary, lambda_step,
                train=False, test=True, strict_imitation=False, new_error=True)
            train_sweep = get_sweep_data("tests/sweeps", train_sweep_entry["datapoint_index"])
            test_sweep = get_sweep_data("tests/sweeps", test_sweep_entry["datapoint_index"])
            error_instances_training = list(map(lambda e: e["instance_id"], train_sweep["error_instances"]))
            error_instances_testing = list(map(lambda e: e["instance_id"], test_sweep["error_instances"]))

            assert(len(set(error_instances_training).intersection(set(error_instances_testing))) == 0)
            lambda_step = lambda_step + 0.05

    def test_bc_cross_entropy_loss(self):
        h1 = LogisticRegression(9, 2)
        optimizer = optim.SGD(h1.parameters(), lr=self.learning_rate, momentum=self.momentum)
        training.train(
            self.n_epochs, h1, optimizer, F.nll_loss, self.partial_training_set, self.testing_set,
            self.batch_size_train, self.batch_size_test)

        h1.eval()

        with torch.no_grad():
            _, _, output = h1(self.testing_set[0][1])

        h1_accuracy = accuracy_score(output.data.max(1)[1].numpy(), self.testing_set[0][2].numpy())

        h2_0 = MLPClassifier(9, 2)
        h2_1 = MLPClassifier(9, 2)

        new_optimizer_0 = optim.SGD(h2_0.parameters(), lr=self.learning_rate, momentum=self.momentum)
        new_optimizer_1 = optim.SGD(h2_0.parameters(), lr=self.learning_rate, momentum=self.momentum)
        bc_cross_entropy_loss_0 = bcloss.BCCrossEntropyLoss(h1, h2_0, 0.0)
        bc_cross_entropy_loss_1 = bcloss.BCCrossEntropyLoss(h1, h2_1, 1.0)
        training.train_compatibility(
            self.n_epochs, h2_0, new_optimizer_0, bc_cross_entropy_loss_0, self.training_set, self.testing_set,
            self.batch_size_train, self.batch_size_test)
        training.train_compatibility(
            self.n_epochs, h2_1, new_optimizer_1, bc_cross_entropy_loss_1, self.training_set, self.testing_set,
            self.batch_size_train, self.batch_size_test)

        evaluation_0 = training.evaluate_model_performance_and_compatibility_on_dataset(
            h1, h2_0, self.testing_set, model_accuracy)
        evaluation_1 = training.evaluate_model_performance_and_compatibility_on_dataset(
            h1, h2_1, self.testing_set, model_accuracy)

        assert(evaluation_0["h2_performance"] >= h1_accuracy)

        # Todo: Need to find a different dataset on which the btc and bec behave as expected
        assert(evaluation_0["h2_performance"] >= evaluation_1["h2_performance"])

    def test_ne_and_si_behaviour_nll_loss(self):
        h1 = LogisticRegression(9, 2)
        optimizer = optim.SGD(h1.parameters(), lr=self.learning_rate, momentum=self.momentum)
        train_counter, test_counter, train_losses, test_losses = training.train(
            self.n_epochs, h1, optimizer, F.nll_loss, self.partial_training_set, self.testing_set,
            self.batch_size_train, self.batch_size_test)

        h1.eval()

        with torch.no_grad():
            _, _, output = h1(self.testing_set[0][1])

        h1_accuracy = accuracy_score(output.data.max(1)[1].numpy(), self.testing_set[0][2].numpy())

        h2 = MLPClassifier(9, 2)

        lambda_c = 0.2

        h2_new_error = copy.deepcopy(h2)
        new_error_loss_kwargs = dict()
        optimizer_kwargs = {"lr": self.learning_rate, "momentum": self.momentum}
        bc_loss = bcloss.BCNLLLoss(h1, h2, lambda_c, **new_error_loss_kwargs)
        optimizer_ne = optim.SGD(h2_new_error.parameters(), **optimizer_kwargs)
        _, _, train_losses_ne, test_losses_ne = training.train_compatibility(
            self.n_epochs, h2, optimizer_ne, bc_loss, self.training_set, self.testing_set,
            self.batch_size_train, self.batch_size_test)

        h2_strict_imitation = copy.deepcopy(h2)
        optimizer_kwargs = {"lr": self.learning_rate, "momentum": self.momentum}
        bc_loss = bcloss.StrictImitationNLLLoss(h1, h2, lambda_c, **new_error_loss_kwargs)
        optimizer_si = optim.SGD(h2_strict_imitation.parameters(), **optimizer_kwargs)
        _, _, train_losses_si, test_losses_si = training.train_compatibility(
            self.n_epochs, h2, optimizer_si, bc_loss, self.training_set, self.testing_set,
            self.batch_size_train, self.batch_size_test)

        diff_train = np.array(train_losses_ne) - np.array(train_losses_si)
        diff_test = np.array(test_losses_ne) - np.array(test_losses_si)

        assert(np.linalg.norm(diff_train) > 1e-3)
        assert(np.linalg.norm(diff_test) > 1e-3)

        training.compatibility_sweep(
            "tests/sweeps", self.n_epochs, h1, h2, self.training_set, self.testing_set,
            self.batch_size_train, self.batch_size_test,
            optim.SGD, {"lr": self.learning_rate, "momentum": self.momentum},
            bcloss.BCNLLLoss, bcloss.StrictImitationNLLLoss,
            lambda_c_stepsize=0.05)

        sweep_summary_file = open("tests/sweeps/sweep_summary.json", "r")
        sweep_summary = json.loads(sweep_summary_file.read())
        lambda_c_values = sorted(list(set(map(lambda i: i["lambda_c"], sweep_summary["data"]))))

        error_diffs_train = []
        error_diffs_test = []
        previous_ne_model = copy.deepcopy(h2)
        previous_si_model = copy.deepcopy(h2)
        ne_model_consecutive_deltas = []
        si_model_consecutive_deltas = []
        ne_train_bec = []
        ne_test_bec = []
        si_train_bec = []
        si_test_bec = []
        ne_train_btc = []
        ne_test_btc = []
        si_train_btc = []
        si_test_btc = []
        for lambda_c in lambda_c_values:
            new_error_model = MLPClassifier(9, 2)
            strict_imitation_model = MLPClassifier(9, 2)

            new_error_model.load_state_dict(torch.load(get_sweep_model_state_filename("tests/sweeps/", lambda_c, new_error=True)))
            strict_imitation_model.load_state_dict(torch.load(get_sweep_model_state_filename("tests/sweeps/", lambda_c, strict_imitation=True)))

            ne_model_delta = []
            for i in range(len(new_error_model.layers)):
                ne_model_delta.append(np.linalg.norm((new_error_model.layers[i].weight.detach() - previous_ne_model.layers[i].weight.detach()).numpy()))

            si_model_delta = []
            for i in range(len(strict_imitation_model.layers)):
                si_model_delta.append(np.linalg.norm((strict_imitation_model.layers[i].weight.detach() - previous_si_model.layers[i].weight.detach()).numpy()))

            ne_model_consecutive_deltas.append(ne_model_delta)
            si_model_consecutive_deltas.append(si_model_delta)

            sweep_entry_ne_train = get_sweep_entry(sweep_summary, lambda_c, train=True, new_error=True)
            sweep_entry_si_train = get_sweep_entry(sweep_summary, lambda_c, train=True, strict_imitation=True)
            sweep_entry_ne_test = get_sweep_entry(sweep_summary, lambda_c, test=True, new_error=True)
            sweep_entry_si_test = get_sweep_entry(sweep_summary, lambda_c, test=True, strict_imitation=True)

            ne_train_p_bec = (sweep_entry_ne_train["bec"], sweep_entry_ne_train["performance"])
            si_train_p_bec = (sweep_entry_si_train["bec"], sweep_entry_ne_train["performance"])
            ne_train_p_btc = (sweep_entry_ne_train["btc"], sweep_entry_ne_train["performance"])
            si_train_p_btc = (sweep_entry_si_train["btc"], sweep_entry_ne_train["performance"])
            ne_test_p_bec = (sweep_entry_ne_test["bec"], sweep_entry_ne_train["performance"])
            si_test_p_bec = (sweep_entry_si_test["bec"], sweep_entry_ne_train["performance"])
            ne_test_p_btc = (sweep_entry_ne_test["btc"], sweep_entry_ne_train["performance"])
            si_test_p_btc = (sweep_entry_si_test["btc"], sweep_entry_ne_train["performance"])

            ne_train_bec.append(ne_train_p_bec)
            ne_test_bec.append(ne_test_p_bec)
            si_train_bec.append(si_train_p_bec)
            si_test_bec.append(si_test_p_bec)
            ne_train_btc.append(ne_train_p_btc)
            ne_test_btc.append(ne_test_p_btc)
            si_train_btc.append(ne_train_p_btc)
            si_test_btc.append(ne_test_p_btc)

            sweep_data_ne_train = get_sweep_data("tests/sweeps/", sweep_entry_ne_train["datapoint_index"])
            sweep_data_si_train = get_sweep_data("tests/sweeps/", sweep_entry_si_train["datapoint_index"])
            sweep_data_ne_test = get_sweep_data("tests/sweeps/", sweep_entry_ne_test["datapoint_index"])
            sweep_data_si_test = get_sweep_data("tests/sweeps/", sweep_entry_si_test["datapoint_index"])

            error_instances_ne_train = sweep_data_ne_train["models_error_overlap"][1]
            error_instances_si_train = sweep_data_si_train["models_error_overlap"][1]
            error_instances_ne_test = sweep_data_ne_test["models_error_overlap"][1]
            error_instances_si_test = sweep_data_si_test["models_error_overlap"][1]

            error_diffs_train.append(len(set(error_instances_ne_train).symmetric_difference(set(error_instances_si_train))))
            error_diffs_test.append(len(set(error_instances_ne_test).symmetric_difference(set(error_instances_si_test))))

            previous_ne_model = new_error_model
            previous_si_model = strict_imitation_model

        assert(max(error_diffs_train) > 0)
        assert(max(error_diffs_test) > 0)

        ne_model_consecutive_delta_norms = list(map(lambda d: np.linalg.norm(np.array(d)), ne_model_consecutive_deltas))
        si_model_consecutive_delta_norms = list(map(lambda d: np.linalg.norm(np.array(d)), si_model_consecutive_deltas))

        assert(len(list(filter(lambda n: n < 1e-6, ne_model_consecutive_delta_norms))) < 2)
        assert(len(list(filter(lambda n: n < 1e-6, si_model_consecutive_delta_norms))) < 2)
