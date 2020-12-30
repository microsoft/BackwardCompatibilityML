# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from backwardcompatibilityml.helpers import training


def compare_models(h1, h2, dataset, performance_metric,
                   get_instance_metadata=None,
                   device="cpu"):
    result = training.evaluate_model_performance_and_compatibility_on_dataset(
        h1, h2, dataset, performance_metric,
        get_instance_metadata=get_instance_metadata, device=device)
    return result
