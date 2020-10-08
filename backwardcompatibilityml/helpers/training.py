# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import backwardcompatibilityml.scores as scores


def train_epoch(epoch, network, optimizer, loss_function, training_set, batch_size_train,
                device="cpu"):
    """
    Trains a model over a single training epoch, with respect to a loss function,
    using an instance of an optimizer.

    (Please note that this is not to be used for training with a
    compatibility loss function.)

    Args:
        network: The model which is undergoing training.
        optimizer: The optimizer instance to use for training.
        loss_function: An instance of the loss function to use for training.
        training_set: The list of training samples as (input, target) pairs.
        batch_size_train: An integer representing batch size of the training set.
        device: A string with values either "cpu" or "cuda" to indicate the
            device that Pytorch is performing training on. By default this
            value is "cpu". But in case your models reside on the GPU, make sure
            to set this to "cuda". This makes sure that the input and target
            tensors are transferred to the GPU during training.
    Returns:
        A list of pairs of the form (training_instance_index, training_loss)
        at regular intervals of 10 training samples.
    """
    log_interval = 10
    train_losses = []
    train_counter = []
    network.train()
    for batch_idx, (data, target) in enumerate(training_set):
        if device != "cpu":
            data = data.to(device)
            target = target.to(device)
        optimizer.zero_grad()
        _, _, output = network(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(training_set) * batch_size_train,
            #     100. * batch_idx / len(training_set), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * batch_size_train) + ((epoch - 1) * len(training_set) * batch_size_train))

    return train_counter, train_losses


def test(network, loss_function, test_set, batch_size_test, device="cpu"):
    """
    Tests a model in a test set using the loss function provided.

    (Please note that this is not to be used for testing with a
    compatibility loss function.)

    Args:
        network: The model which is undergoing testing.
        loss_function: An instance of the loss function to use for training.
        test_set: The list of testing samples as (input, target) pairs.
        batch_size_test: An integer representing the batch size of the test set.
        device: A string with values either "cpu" or "cuda" to indicate the
            device that Pytorch is performing training on. By default this
            value is "cpu". But in case your models reside on the GPU, make sure
            to set this to "cuda". This makes sure that the input and target
            tensors are transferred to the GPU during training.
    Returns:
        Returns a list of test loses.
    """
    network.eval()
    test_losses = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_set:
            if device != "cpu":
                data = data.to(device)
                target = target.to(device)
            _, _, output = network(data)
            test_loss += loss_function(output, target, reduction="sum").item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_set) * batch_size_test
        test_losses.append(test_loss)
        # print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(test_set) * batch_size_test,
        #     100. * correct / (len(test_set) * batch_size_test)))

    return test_losses


def train(number_of_epochs, network, optimizer, loss_function,
          training_set, test_set, batch_size_train, batch_size_test,
          device="cpu"):
    """
    Trains a model with respect to a loss function, using an instance
    of an optimizer.

    (Please note that this is not to be used for training with a
    compatibility loss function.)

    Args:
        network: The model which is undergoing training.
        number_of_epochs: Number of epochs of training.
        optimizer: The optimizer instance to use for training.
        loss_function: An instance of the loss function to use for training.
        training_set: The list of training samples as (input, target) pairs.
        test_set: The list of testing samples as (input, target) pairs.
        batch_size_train: An integer representing batch size of the training set.
        batch_size_test: An integer representing the batch size of the test set.
        device: A string with values either "cpu" or "cuda" to indicate the
            device that Pytorch is performing training on. By default this
            value is "cpu". But in case your models reside on the GPU, make sure
            to set this to "cuda". This makes sure that the input and target
            tensors are transferred to the GPU during training.
    Returns:
        |  Returns four lists
        |  **train_counter** - The index of a training samples at which training losses were logged.
        |  **test_counter** - The index of testing samples at which testing losses were logged.
        |  **train_losses** - The list of logged training losses.
        |  **test_losses** - The list of logged testing losses.
    """
    train_counter = []
    test_counter = [i * len(training_set) * batch_size_train for i in range(number_of_epochs + 1)]
    train_losses = []
    test_losses = []
    test_losses_run = test(
        network, loss_function, test_set, batch_size_test, device=device)
    test_losses = test_losses + test_losses_run
    for epoch in range(1, number_of_epochs + 1):
        train_counter_run, train_losses_run = train_epoch(
            epoch, network, optimizer, loss_function, training_set, batch_size_train,
            device=device)
        train_counter = train_counter + train_counter_run
        train_losses = train_losses + train_losses_run
        test_losses_run = test(
            network, loss_function, test_set, batch_size_test, device=device)
        test_losses = test_losses + test_losses_run

    return train_counter, test_counter, train_losses, test_losses


def train_compatibility_epoch(epoch, h2, optimizer, loss_function, training_set, batch_size_train,
                              device="cpu"):
    """
    Trains a new model using the instance compatibility loss function provided,
    over a single epoch. The compatibility loss function instnace may be either
    a New Error or Strict Imitation type loss function.

    Args:
        epoch: The integer index of the training epoch being run.
        h2: The model which is undergoing training / updating.
        optimizer: The optimizer instance to use for training.
        loss_function: An instance of a compatibility loss function.
        training_set: The list of training samples as (input, target) pairs.
        batch_size_train: An integer representing batch size of the training set.
        device: A string with values either "cpu" or "cuda" to indicate the
            device that Pytorch is performing training on. By default this
            value is "cpu". But in case your models reside on the GPU, make sure
            to set this to "cuda". This makes sure that the input and target
            tensors are transferred to the GPU during training.
    Returns:
        A list of pairs of the form (training_instance_index, training_loss)
        at regular intervals of 10 training samples.
    """
    log_interval = 10
    train_losses = []
    train_counter = []
    h2.train()
    for batch_idx, (data, target) in enumerate(training_set):
        if device != "cpu":
            data = data.to(device)
            target = target.to(device)
        optimizer.zero_grad()
        loss = loss_function(data, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(training_set) * batch_size_train,
            #     100. * batch_idx / len(training_set), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(training_set) * batch_size_train))

    return train_counter, train_losses


def test_compatibility(h2, loss_function, test_set, batch_size_test, device="cpu"):
    """
    Tests a model in a test set using the backward compatibility loss function provided.

    Args:
        h2: The model which is undergoing training / updating.
        loss_function: An instance of a compatibility loss function.
        test_set: The list of testing samples as (input, target) pairs.
        batch_size_test: An integer representing the batch size of the test set.
        device: A string with values either "cpu" or "cuda" to indicate the
            device that Pytorch is performing training on. By default this
            value is "cpu". But in case your models reside on the GPU, make sure
            to set this to "cuda". This makes sure that the input and target
            tensors are transferred to the GPU during training.
    Returns:
        Returns a list of test loses.
    """
    h2.eval()
    test_losses = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_set:
            if device != "cpu":
                data = data.to(device)
                target = target.to(device)
            _, _, output = h2(data)
            test_loss += loss_function(data, target, reduction="sum").item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_set) * batch_size_test
        test_losses.append(test_loss)
        # print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(test_set) * batch_size_test,
        #     100. * correct / (len(test_set) * batch_size_test)))

    return test_losses


def train_compatibility(number_of_epochs, h2, optimizer, loss_function,
                        training_set, test_set, batch_size_train, batch_size_test,
                        device="cpu"):
    """
    Trains a new model with respect to an existing model using the
    compatibility loss function provided. The compatibility loss function
    may be either a New Error or Strict Imitation type loss function.

    Args:
        h2: The model which is undergoing training / updating.
        number_of_epochs: Number of epochs of training.
        loss_function: An instance of a compatibility loss function.
        optimizer: The optimizer instance to use for training.
        training_set: The list of training samples as (input, target) pairs.
        test_set: The list of testing samples as (input, target) pairs.
        batch_size_train: An integer representing batch size of the training set.
        batch_size_test: An integer representing the batch size of the test set.
        device: A string with values either "cpu" or "cuda" to indicate the
            device that Pytorch is performing training on. By default this
            value is "cpu". But in case your models reside on the GPU, make sure
            to set this to "cuda". This makes sure that the input and target
            tensors are transferred to the GPU during training.
    Returns:
        |  Returns four lists
        |  **train_counter** - The index of a training samples at which training losses were logged.
        |  **test_counter** - The index of testing samples at which testing losses were logged.
        |  **train_losses** - The list of logged training losses.
        |  **test_losses** - The list of logged testing losses.
    """
    train_counter = []
    test_counter = [i * len(training_set) * batch_size_train for i in range(number_of_epochs + 1)]
    train_losses = []
    test_losses = []
    test_losses_run = test_compatibility(
        h2, loss_function, test_set, batch_size_test, device=device)
    test_losses = test_losses + test_losses_run
    for epoch in range(1, number_of_epochs + 1):
        train_counter_run, train_losses_run = train_compatibility_epoch(
            epoch, h2, optimizer, loss_function, training_set, batch_size_train,
            device=device)
        train_counter = train_counter + train_counter_run
        train_losses = train_losses + train_losses_run
        test_losses_run = test_compatibility(
            h2, loss_function, test_set, batch_size_test, device=device)
        test_losses = test_losses + test_losses_run

    return train_counter, test_counter, train_losses, test_losses


def get_error_instance_indices(model, batched_evaluation_data, batched_evaluation_target,
                               device="cpu"):
    """
    Return the list of indices of instances from batched_evaluation_data on which the
    model prediction differs from the ground truth in batched_evaluation_target.

    Args:
        model: The model being evaluated.
        batched_evaluation_data: A single batch of input data to be passed to our model.
        batched_evaluation_target: A single batch of the corresponding output targets.
        device: A string with values either "cpu" or "cuda" to indicate the
            device that Pytorch is performing training on. By default this
            value is "cpu". But in case your models reside on the GPU, make sure
            to set this to "cuda". This makes sure that the input and target
            tensors are transferred to the GPU during training.
    Returns:
        A list of indices of the instances within the batched data, for which the
        model did not match the expected target.
    """
    with torch.no_grad():
        if device != "cpu":
            batched_evaluation_data = batched_evaluation_data.to(device)
            batched_evaluation_target = batched_evaluation_target.to(device)
        _, _, model_output_logsoftmax = model(batched_evaluation_data)
        model_diff = (torch.argmax(model_output_logsoftmax, 1) - batched_evaluation_target).float()

        return model_diff.nonzero().view(-1).tolist()


def get_normalized_model_error_overlap(h1, h2, batched_evaluation_data, batched_evaluation_target,
                                       device="cpu"):
    """
    Return the fraction of errors of each model and the fraction of errors common to both models,
    all with respect to the total number of errors of each model.

    Args:
        h1: Reference Pytorch model.
        h2: The model being compared to h1.
        batched_evaluation_data: A single batch of input data to be passed to our model.
        batched_evaluation_target: A single batch of the corresponding output targets.
        device: A string with values either "cpu" or "cuda" to indicate the
            device that Pytorch is performing training on. By default this
            value is "cpu". But in case your models reside on the GPU, make sure
            to set this to "cuda". This makes sure that the input and target
            tensors are transferred to the GPU during training.
    Returns:
        |  If there are any errors at all it returns a triple of the form 
            proportion_of_error_due_to_h1, 
            proportion_of_error_due_to_h2, 
            proportion_of_error_due_to_h1_and_h2
        |  If there are no errors at all it returns the triple 
            0, 0, 0
    """
    h1_error_indices = get_error_instance_indices(
        h1, batched_evaluation_data, batched_evaluation_target, device=device)
    h2_error_indices = get_error_instance_indices(
        h2, batched_evaluation_data, batched_evaluation_target, device=device)
    h1_size = len(h1_error_indices)
    h2_size = len(h2_error_indices)
    h1_and_h2_size = len(set(h1_error_indices).intersection(set(h2_error_indices)))
    total = h1_size + h2_size - h1_and_h2_size

    if total > 0:
        return (h1_size / total), (h2_size / total), (h1_and_h2_size / total)

    return 0, 0, 0


def get_error_fraction_by_class(model, batched_evaluation_data, batched_evaluation_target,
                                device="cpu"):
    """
    Return the fraction of errors of the model by class.

    Args:
        model: The model being evaluated.
        batched_evaluation_data: A single batch of input data to be passed to our model.
        batched_evaluation_target: A single batch of the corresponding output targets.
        device: A string with values either "cpu" or "cuda" to indicate the
            device that Pytorch is performing training on. By default this
            value is "cpu". But in case your models reside on the GPU, make sure
            to set this to "cuda". This makes sure that the input and target
            tensors are transferred to the GPU during training.
    Returns:
        A dictionary of key / value pairs, where the key is the output class
        and the value is the fraction of misclassification errors of the
        model within that class.
    """
    with torch.no_grad():
        if device != "cpu":
            batched_evaluation_data = batched_evaluation_data.to(device)
            batched_evaluation_target = batched_evaluation_target.to(device)
        _, _, model_output_logsoftmax = model(batched_evaluation_data)
        model_diff = (torch.argmax(model_output_logsoftmax, 1) - batched_evaluation_target).float()
        model_errors = (model_diff != 0)
        target_error_classes = batched_evaluation_target[model_errors].view(-1).tolist()
        class_error_count = {}
        total_errors = len(target_error_classes)
        for class_label in set(batched_evaluation_target.view(-1).tolist()):
            class_error_count[class_label] = 0
        for error_class in target_error_classes:
            class_error_count[error_class] = class_error_count[error_class] + 1

        if total_errors > 0:
            for class_label, error_count in class_error_count.items():
                class_error_count[class_label] = error_count / total_errors

        return class_error_count


def compatibility_scores(h1, h2, dataset, device="cpu"):
    """
    Args:
        h1: Reference Pytorch model.
        h2: The model being compared to h1.
        dataset: Data in the form of a list of batches of input/target pairs.
        device: A string with values either "cpu" or "cuda" to indicate the
            device that Pytorch is performing training on. By default this
            value is "cpu". But in case your models reside on the GPU, make sure
            to set this to "cuda". This makes sure that the input and target
            tensors are transferred to the GPU during training.
    Returns:
        A pair consisting of **btc_dataset** - the average trust compatibility
        score over all batches, and **bec_dataset** - the average error 
        compatibility score over all batches.
    """
    number_of_batches = len(dataset)
    with torch.no_grad():
        btc_dataset = 0
        bec_dataset = 0
        for data, target in dataset:
            if device != "cpu":
                data = data.to(device)
                target = target.to(device)
            _, _, h1_output_logsoftmax = h1(data)
            _, _, h2_output_logsoftmax = h2(data)
            h1_output_labels = torch.argmax(h1_output_logsoftmax, 1)
            h2_output_labels = torch.argmax(h2_output_logsoftmax, 1)
            btc = scores.trust_compatibility_score(h1_output_labels, h2_output_labels, target)
            bec = scores.error_compatibility_score(h1_output_labels, h2_output_labels, target)
            btc_dataset += btc
            bec_dataset += bec

        btc_dataset /= number_of_batches
        bec_dataset /= number_of_batches

    return btc_dataset, bec_dataset


def model_accuracy(model, dataset, device="cpu"):
    number_of_batches = len(dataset)
    model_performance = 0
    with torch.no_grad():
        for data, target in dataset:
            if device != "cpu":
                data = data.to(device)
                target = target.to(device)
            _, _, output_logsoftmax = model(data)
            output_labels = torch.argmax(output_logsoftmax, 1)
            if device != "cpu":
                output_labels = output_labels.cpu()
                target = target.cpu()
            performance = accuracy_score(output_labels.numpy(), target.numpy())
            model_performance += performance
            # _clean_from_gpu([data, target])

        model_performance /= number_of_batches

    return model_performance


def evaluate_model_performance_and_compatibility_on_dataset(h1, h2, dataset, performance_metric=None,
                                                            device="cpu"):
    """
    Args:
        h1: The reference model being used.
        h2: The model being traind / updated.
        performance_metric: Optional performance metric to be used when evaluating the model.
            If not specified then accuracy is used.
        device: A string with values either "cpu" or "cuda" to indicate the
            device that Pytorch is performing training on. By default this
            value is "cpu". But in case your models reside on the GPU, make sure
            to set this to "cuda". This makes sure that the input and target
            tensors are transferred to the GPU during training.
    Returns:
        A dictionary containing the models error overlap between h1 and h2,
        the error fraction by class of the model h2,
        the trust compatibility score of h2 with respect to h1, and
        the error compatibility score of h2 with respect to h1.
    """
    number_of_batches = len(dataset)
    h1_dataset_error_fraction = 0
    h2_dataset_error_fraction = 0
    h1_and_h2_dataset_error_fraction = 0
    h2_dataset_error_fraction_by_class = {}
    classes = set()
    for data, target in dataset:
        classes = classes.union(target.tolist())
        h1_error_fraction, h2_error_fraction, h1_and_h2_error_fraction =\
            get_normalized_model_error_overlap(h1, h2, data, target, device=device)
        h2_error_fraction_by_class =\
            get_error_fraction_by_class(h2, data, target, device=device)
        h1_dataset_error_fraction += h1_error_fraction
        h2_dataset_error_fraction += h2_error_fraction
        h1_and_h2_dataset_error_fraction += h1_and_h2_error_fraction
        for class_label, error_fraction in h2_error_fraction_by_class.items():
            if class_label in h2_dataset_error_fraction_by_class:
                h2_dataset_error_fraction_by_class[class_label] += error_fraction
            else:
                h2_dataset_error_fraction_by_class[class_label] = error_fraction

    h1_dataset_error_fraction /= number_of_batches
    h2_dataset_error_fraction /= number_of_batches
    h1_and_h2_dataset_error_fraction /= number_of_batches

    for class_label, error_fraction in h2_dataset_error_fraction_by_class.items():
        h2_dataset_error_fraction_by_class[class_label] /= number_of_batches

    h2_ds_error_fraction_by_class = []
    for class_label, error_fraction in h2_dataset_error_fraction_by_class.items():
        h2_ds_error_fraction_by_class.append({
            "class": class_label,
            "incompatibleFraction": error_fraction
        })

    if performance_metric is not None:
        h2_performance = performance_metric(h2, dataset)
    else:
        h2_performance = 0
        with torch.no_grad():
            for data, target in dataset:
                if device != "cpu":
                    data = data.to(device)
                    target = target.to(device)
                _, _, output_logsoftmax = h2(data)
                output_labels = torch.argmax(output_logsoftmax, 1)
                if device != "cpu":
                    output_labels = output_labels.cpu()
                    target = target.cpu()
                performance = accuracy_score(output_labels.numpy(), target.numpy())
                h2_performance += performance
                # _clean_from_gpu([data, target])

            h2_performance /= number_of_batches

    btc, bec = compatibility_scores(h1, h2, dataset, device=device)

    return {
        "models_error_overlap": [
            h1_dataset_error_fraction,
            h2_dataset_error_fraction,
            h1_and_h2_dataset_error_fraction
        ],

        "h2_error_fraction_by_class": h2_ds_error_fraction_by_class,
        "h2_performance": h2_performance,
        "btc": btc,
        "bec": bec
    }


def evaluate_model_performance_and_compatibility(h1, h2, training_set, test_set, performance_metric=None,
                                                 device="cpu"):
    """
    Calculate the error overlap of h1 and h2 on a batched dataset.
    Calculate the h2 model error fraction by class on a batched dataset.

    Args:
        h1: The reference model being used.
        h2: The model being traind / updated.
        performance_metric: Optional performance metric to be used when evaluating the model.
            If not specified then accuracy is used.
        training_set: The list of batched training samples as (input, target) pairs.
        test_set: The list of batched testing samples as (input, target) pairs.
        device: A string with values either "cpu" or "cuda" to indicate the
            device that Pytorch is performing training on. By default this
            value is "cpu". But in case your models reside on the GPU, make sure
            to set this to "cuda". This makes sure that the input and target
            tensors are transferred to the GPU during training.
    Returns:
        A dictionary containing the results of the model performance and evaluation
        performed on the training and the testing sets separately.
    """
    training_set_performance_and_compatibility =\
        evaluate_model_performance_and_compatibility_on_dataset(
            h1, h2, training_set, performance_metric=performance_metric,
            device=device)
    testing_set_performance_and_compatibility =\
        evaluate_model_performance_and_compatibility_on_dataset(
            h1, h2, test_set, performance_metric=performance_metric,
            device=device)

    return {
        "training": training_set_performance_and_compatibility,
        "test": testing_set_performance_and_compatibility
    }


def train_new_error(h1, h2, number_of_epochs,
                    training_set, test_set, batch_size_train, batch_size_test,
                    OptimizerClass, optimizer_kwargs,
                    NewErrorLossClass,
                    lambda_c, new_error_loss_kwargs=None, device="cpu"):
    """
    Args:
        h1: Reference Pytorch model.
        h2: The model which is undergoing training / updating.
        number_of_epochs: Number of epochs of training.
        training_set: The list of training samples as (input, target) pairs.
        test_set: The list of testing samples as (input, target) pairs.
        batch_size_train: An integer representing batch size of the training set.
        batch_size_test: An integer representing the batch size of the test set.
        OptimizerClass: The class to instantiate an optimizer from for training.
        optimizer_kwargs: A dictionary of the keyword arguments to be used to
            instantiate the optimizer.
        NewErrorLossClass: The class of the New Error style loss function to
            be instantiated and used to perform compatibility constrained
            training of our model h2.
        lambda_c: The regularization parameter to be used when calibrating the
            degree of compatibility to enforce while training.
        device: A string with values either "cpu" or "cuda" to indicate the
            device that Pytorch is performing training on. By default this
            value is "cpu". But in case your models reside on the GPU, make sure
            to set this to "cuda". This makes sure that the input and target
            tensors are transferred to the GPU during training.
    """
    bc_loss = NewErrorLossClass(h1, h2, lambda_c, **new_error_loss_kwargs)
    new_optimizer = OptimizerClass(h2.parameters(), **optimizer_kwargs)
    _, _, _, _ = train_compatibility(
        number_of_epochs, h2, new_optimizer, bc_loss, training_set, test_set,
        batch_size_train, batch_size_test, device=device)


def train_strict_imitation(h1, h2, number_of_epochs,
                           training_set, test_set, batch_size_train, batch_size_test,
                           OptimizerClass, optimizer_kwargs,
                           StrictImitationLossClass,
                           lambda_c, strict_imitation_loss_kwargs=None, device="cpu"):
    """
    Args:
        h1: Reference Pytorch model.
        h2: The model which is undergoing training / updating.
        number_of_epochs: Number of epochs of training.
        training_set: The list of training samples as (input, target) pairs.
        test_set: The list of testing samples as (input, target) pairs.
        batch_size_train: An integer representing batch size of the training set.
        batch_size_test: An integer representing the batch size of the test set.
        OptimizerClass: The class to instantiate an optimizer from for training.
        optimizer_kwargs: A dictionary of the keyword arguments to be used to
            instantiate the optimizer.
        StrictImitationLossClass: The class of the Strict Imitation style loss
            function to be instantiated and used to perform compatibility
            constrained training of our model h2.
        lambda_c: The regularization parameter to be used when calibrating the
            degree of compatibility to enforce while training.
        device: A string with values either "cpu" or "cuda" to indicate the
            device that Pytorch is performing training on. By default this
            value is "cpu". But in case your models reside on the GPU, make sure
            to set this to "cuda". This makes sure that the input and target
            tensors are transferred to the GPU during training.
    """
    si_loss = StrictImitationLossClass(h1, h2, lambda_c, **strict_imitation_loss_kwargs)
    new_optimizer = OptimizerClass(h2.parameters(), **optimizer_kwargs)
    _, _, _, _ = train_compatibility(
        number_of_epochs, h2, new_optimizer, si_loss, training_set, test_set,
        batch_size_train, batch_size_test, device=device)


def compatibility_sweep(sweeps_folder_path, number_of_epochs, h1, h2,
                        training_set, test_set, batch_size_train, batch_size_test,
                        OptimizerClass, optimizer_kwargs,
                        NewErrorLossClass, StrictImitationLossClass,
                        performance_metric=None,
                        lambda_c_stepsize=0.25, percent_complete_queue=None,
                        new_error_loss_kwargs=None,
                        strict_imitation_loss_kwargs=None,
                        device="cpu"):
    """
    This function trains a new model using the backward compatibility loss function
    BCNLLLoss with respect to an existing model. It does this for each value of
    lambda_c betweek 0 and 1 at the specified step sizes. It saves the newly
    trained models in the specified folder.

    Args:
        sweeps_folder_path: A string value representing the full path of the
            folder wehre the result of the compatibility sweep is to be stored.
        number_of_epochs: The number of training epochs to use on each sweep.
        h1: The reference model being used.
        h2: The new model being traind / updated.
        training_set: The list of training samples as (input, target) pairs.
        test_set: The list of testing samples as (input, target) pairs.
        batch_size_train: An integer representing batch size of the training set.
        batch_size_test: An integer representing the batch size of the test set.
        OptimizerClass: The class to instantiate an optimizer from for training.
        optimizer_kwargs: A dictionary of the keyword arguments to be used to
            instantiate the optimizer.
        NewErrorLossClass: The class of the New Error style loss function to
            be instantiated and used to perform compatibility constrained
            training of our model h2.
        StrictImitationLossClass: The class of the Strict Imitation style loss
            function to be instantiated and used to perform compatibility
            constrained training of our model h2.
        performance_metric: Optional performance metric to be used when evaluating the model.
            If not specified then accuracy is used.
        lambda_c_stepsize: The increments of lambda_c to use as we sweep the parameter
            space between 0.0 and 1.0.
        percent_complete_queue: Optional thread safe queue to use for logging the
            status of the sweep in terms of the percentage complete.
        device: A string with values either "cpu" or "cuda" to indicate the
            device that Pytorch is performing training on. By default this
            value is "cpu". But in case your models reside on the GPU, make sure
            to set this to "cuda". This makes sure that the input and target
            tensors are transferred to the GPU during training.
    """
    h1.eval()
    number_of_trainings = 4 * len(np.arange(0.0, 1.0 + (lambda_c_stepsize / 2), lambda_c_stepsize))
    if percent_complete_queue is not None:
        percent_complete_queue.put(0.0)

    if new_error_loss_kwargs is None:
        new_error_loss_kwargs = dict()

    if strict_imitation_loss_kwargs is None:
        strict_imitation_loss_kwargs = dict()

    sweep_summary_data = []
    datapoint_index = 0
    for lambda_c in np.arange(0.0, 1.0 + (lambda_c_stepsize / 2), lambda_c_stepsize):
        h2_new_error = copy.deepcopy(h2)
        train_new_error(
            h1, h2_new_error, number_of_epochs,
            training_set, test_set, batch_size_train, batch_size_test,
            OptimizerClass, optimizer_kwargs, NewErrorLossClass, lambda_c,
            new_error_loss_kwargs=new_error_loss_kwargs,
            device=device)
        h2_new_error.eval()
        torch.save(h2_new_error.state_dict(), f"{sweeps_folder_path}/{lambda_c}-model-new-error.state")

        training_set_performance_and_compatibility =\
            evaluate_model_performance_and_compatibility_on_dataset(
                h1, h2_new_error, training_set, performance_metric=performance_metric,
                device=device)
        training_set_performance_and_compatibility["lambda_c"] = lambda_c
        training_set_performance_and_compatibility["training"] = True
        training_set_performance_and_compatibility["testing"] = False
        training_set_performance_and_compatibility["datapoint_index"] = datapoint_index
        sweep_summary_data.append({
            "datapoint_index": datapoint_index,
            "lambda_c": lambda_c,
            "training": True,
            "testing": False,
            "new-error": True,
            "strict-imitation": False,
            "performance": training_set_performance_and_compatibility["h2_performance"],
            "btc": training_set_performance_and_compatibility["btc"],
            "bec": training_set_performance_and_compatibility["bec"]
        })
        training_evaluation_data = json.dumps(training_set_performance_and_compatibility)
        training_evaluation_data_file = open(f"{sweeps_folder_path}/{datapoint_index}-evaluation-data.json", "w")
        training_evaluation_data_file.write(training_evaluation_data)
        training_evaluation_data_file.close()
        datapoint_index += 1

        testing_set_performance_and_compatibility =\
            evaluate_model_performance_and_compatibility_on_dataset(
                h1, h2_new_error, test_set, performance_metric=performance_metric,
                device=device)
        testing_set_performance_and_compatibility["lambda_c"] = lambda_c
        testing_set_performance_and_compatibility["training"] = False
        testing_set_performance_and_compatibility["testing"] = True
        testing_set_performance_and_compatibility["datapoint_index"] = datapoint_index
        sweep_summary_data.append({
            "datapoint_index": datapoint_index,
            "lambda_c": lambda_c,
            "training": False,
            "testing": True,
            "new-error": True,
            "strict-imitation": False,
            "performance": testing_set_performance_and_compatibility["h2_performance"],
            "btc": testing_set_performance_and_compatibility["btc"],
            "bec": testing_set_performance_and_compatibility["bec"]
        })
        testing_evaluation_data = json.dumps(testing_set_performance_and_compatibility)
        testing_evaluation_data_file = open(f"{sweeps_folder_path}/{datapoint_index}-evaluation-data.json", "w")
        testing_evaluation_data_file.write(testing_evaluation_data)
        testing_evaluation_data_file.close()
        datapoint_index += 1

        h2_strict_imitation = copy.deepcopy(h2)
        train_strict_imitation(
            h1, h2_strict_imitation, number_of_epochs,
            training_set, test_set, batch_size_train, batch_size_test,
            OptimizerClass, optimizer_kwargs, StrictImitationLossClass, lambda_c,
            strict_imitation_loss_kwargs=strict_imitation_loss_kwargs,
            device=device)
        h2_strict_imitation.eval()
        torch.save(h2_strict_imitation.state_dict(), f"{sweeps_folder_path}/{lambda_c}-model-strict-imitation.state")

        training_set_performance_and_compatibility =\
            evaluate_model_performance_and_compatibility_on_dataset(
                h1, h2_strict_imitation, training_set, performance_metric=performance_metric,
                device=device)
        training_set_performance_and_compatibility["lambda_c"] = lambda_c
        training_set_performance_and_compatibility["training"] = True
        training_set_performance_and_compatibility["testing"] = False
        training_set_performance_and_compatibility["datapoint_index"] = datapoint_index
        sweep_summary_data.append({
            "datapoint_index": datapoint_index,
            "lambda_c": lambda_c,
            "training": True,
            "testing": False,
            "new-error": False,
            "strict-imitation": True,
            "performance": training_set_performance_and_compatibility["h2_performance"],
            "btc": training_set_performance_and_compatibility["btc"],
            "bec": training_set_performance_and_compatibility["bec"]
        })
        training_evaluation_data = json.dumps(training_set_performance_and_compatibility)
        training_evaluation_data_file = open(f"{sweeps_folder_path}/{datapoint_index}-evaluation-data.json", "w")
        training_evaluation_data_file.write(training_evaluation_data)
        training_evaluation_data_file.close()
        datapoint_index += 1

        testing_set_performance_and_compatibility =\
            evaluate_model_performance_and_compatibility_on_dataset(
                h1, h2_new_error, test_set, performance_metric=performance_metric,
                device=device)
        testing_set_performance_and_compatibility["lambda_c"] = lambda_c
        testing_set_performance_and_compatibility["training"] = False
        testing_set_performance_and_compatibility["testing"] = True
        testing_set_performance_and_compatibility["datapoint_index"] = datapoint_index
        sweep_summary_data.append({
            "datapoint_index": datapoint_index,
            "lambda_c": lambda_c,
            "training": False,
            "testing": True,
            "new-error": False,
            "strict-imitation": True,
            "performance": testing_set_performance_and_compatibility["h2_performance"],
            "btc": testing_set_performance_and_compatibility["btc"],
            "bec": testing_set_performance_and_compatibility["bec"]
        })
        testing_evaluation_data = json.dumps(testing_set_performance_and_compatibility)
        testing_evaluation_data_file = open(f"{sweeps_folder_path}/{datapoint_index}-evaluation-data.json", "w")
        testing_evaluation_data_file.write(testing_evaluation_data)
        testing_evaluation_data_file.close()
        datapoint_index += 1

        if percent_complete_queue is not None:
            percent_complete_queue.put((datapoint_index) / number_of_trainings)

    sweep_summary = {
        "data": sweep_summary_data,
        "h1_performance": model_accuracy(h1, test_set)
    }

    sweep_summary_data = json.dumps(sweep_summary)
    sweep_summary_data_file = open(f"{sweeps_folder_path}/sweep_summary.json", "w")
    sweep_summary_data_file.write(sweep_summary_data)
    sweep_summary_data_file.close()
