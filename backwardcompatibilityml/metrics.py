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
