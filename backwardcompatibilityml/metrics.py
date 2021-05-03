# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from sklearn.metrics import accuracy_score


def model_accuracy(model, dataset, device="cpu"):
    model_performance = 0
    number_of_batches = len(dataset)
    with torch.no_grad():
        for batch_ids, data, target in dataset:
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


def model_accuracy_by_class(model, dataset, device="cpu"):
    accuracy_by_class = dict()
    target_class_count = dict()
    with torch.no_grad():
        for batch_ids, data, target in dataset:
            if device != "cpu":
                data = data.to(device)
                target = target.to(device)
            _, _, output_logsoftmax = model(data)
            output_labels = torch.argmax(output_logsoftmax, 1)
            if len(accuracy_by_class.keys()) == 0:
                class_count = output_logsoftmax.size(1)
                for class_key in range(class_count):
                    accuracy_by_class[class_key] = 0.0
                    target_class_count[class_key] = 0.0

            output_labels = output_labels.cpu()
            target = target.cpu()
            for i, target_label in enumerate(target):
                target_label = target_label.item()
                target_class_count[target_label] = target_class_count[target_label] + 1
                if target_label == output_labels[i]:
                    accuracy_by_class[target_label] = accuracy_by_class[target_label] + 1

    model_class_accuracy_list = []
    for class_label, accuracy_count in accuracy_by_class.items():
        model_class_accuracy_list.append({
            "class": class_label,
            "accuracy": accuracy_by_class[class_label] / target_class_count[class_label]})

    return model_class_accuracy_list
