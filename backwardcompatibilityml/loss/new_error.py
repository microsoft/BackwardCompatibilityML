# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from backwardcompatibilityml.helpers import utils


class BCNLLLoss(nn.Module):
    """
    Backward Compatibility Negative Log Likelihood Loss

    This class implements the backward compatibility loss function
    with the underlying loss function being the Negative Log Likelihood
    loss.

    Example usage:
        h1 = MyModel()
        ... train h1 ...
        h1.eval() (it is important that h1 be put in evaluation mode)

        lambda_c = 0.5 (regularization parameter)
        h2 = MyNewModel() (this may be the same model type as MyModel)
        bcloss = BCNLLLoss(h1, h2, lambda_c)

        for x, y in training_data:
            loss = bcloss(x, y)
            loss.backward()

        Note that we pass in the input and the target directly to the
        bcloss function instance. It calculates the outputs of h1 and h2
        internally.

    Args:
        h1: Our reference model which we would like to be compatible with.
        h2: Our new model which will be the updated model.
        lambda_c: A float between 0.0 and 1.0, which is a regularization
            parameter that determines how much we want to penalize model h2
            for being incompatible with h1. Lower values panalize less and
            higher values penalize more.
    """

    def __init__(self, h1, h2, lambda_c, **kwargs):
        super(BCNLLLoss, self).__init__()
        self.h1 = h1
        self.h2 = h2
        self.loss = F.nll_loss
        self.lambda_c = lambda_c

    def forward(self, x, y, reduction="mean"):
        with torch.no_grad():
            _, _, h1_output_logsoftmax = self.h1(x)
        h1_diff = (torch.argmax(h1_output_logsoftmax, 1) - y).float()
        h1_correct = (h1_diff == 0)
        x_support = x[h1_correct]
        y_support = y[h1_correct]
        _, _, h2_output_logsoftmax = self.h2(x)

        # Calculate the dissonance, using instances where h1 makes
        # the correct prediction.
        dissonance = 0.0
        if (h1_diff == 0.0).sum(dim=0).item() > 0:
            _, _, h2_support_output_logsoftmax = self.h2(x_support)
            dissonance = self.loss(h2_support_output_logsoftmax, y_support)

        base_loss = self.loss(h2_output_logsoftmax, y, reduction=reduction)
        new_error_loss = base_loss + self.lambda_c * dissonance
        return new_error_loss


class BCCrossEntropyLoss(nn.Module):
    """
    Backward Compatibility Cross Entropy Loss

    This class implements the backward compatibility loss function
    with the underlying loss function being the Cross Entropy loss.

    Example usage:
        h1 = MyModel()
        ... train h1 ...
        h1.eval() (it is important that h1 be put in evaluation mode)

        lambda_c = 0.5 (regularization parameter)
        h2 = MyNewModel() (this may be the same model type as MyModel)
        bcloss = BCCrossEntropyLoss(h1, h2, lambda_c)

        for x, y in training_data:
            loss = bcloss(x, y)
            loss.backward()

        Note that we pass in the input and the target directly to the
        bcloss function instance. It calculates the outputs of h1 and h2
        internally.

    Args:
        h1: Our reference model which we would like to be compatible with.
        h2: Our new model which will be the updated model.
        lambda_c: A float between 0.0 and 1.0, which is a regularization
            parameter that determines how much we want to penalize model h2
            for being incompatible with h1. Lower values panalize less and
            higher values penalize more.
    """

    def __init__(self, h1, h2, lambda_c, **kwargs):
        super(BCCrossEntropyLoss, self).__init__()
        self.h1 = h1
        self.h2 = h2
        self.loss = F.cross_entropy
        self.lambda_c = lambda_c

    def dissonance(self, h2_output_logit, target_labels):
        cross_entropy_loss = F.cross_entropy(h2_output_logit, target_labels)
        return cross_entropy_loss

    def forward(self, x, y, reduction="mean"):
        with torch.no_grad():
            h1_output_logit, h1_output_softmax, h1_output_logsoftmax = self.h1(x)
        h1_diff = (torch.argmax(h1_output_logsoftmax, 1) - y).float()
        h1_correct = (h1_diff == 0)
        x_support = x[h1_correct]
        y_support = y[h1_correct]
        h2_output_logit, h2_output_softmax, h2_output_logsoftmax = self.h2(x)

        dissonance = 0.0
        if (h1_diff == 0.0).sum(dim=0).item() > 0:
            h2_support_output_logit, _, _ = self.h2(x_support)
            dissonance = self.dissonance(h2_support_output_logit, y_support)

        base_loss = self.loss(h2_output_logit, y, reduction=reduction)
        new_error_loss = base_loss + self.lambda_c * dissonance
        return new_error_loss


class BCBinaryCrossEntropyLoss(nn.Module):
    """
    Backward Compatibility Cross Entropy Loss

    This class implements the backward compatibility loss function
    with the underlying loss function being the Cross Entropy loss.

    Example usage:
        h1 = MyModel()
        ... train h1 ...
        h1.eval() (it is important that h1 be put in evaluation mode)

        lambda_c = 0.5 (regularization parameter)
        h2 = MyNewModel() (this may be the same model type as MyModel)
        bcloss = BCBinaryCrossEntropyLoss(h1, h2, lambda_c)

        for x, y in training_data:
            loss = bcloss(x, y)
            loss.backward()

        Note that we pass in the input and the target directly to the
        bcloss function instance. It calculates the outputs of h1 and h2
        internally.

    Args:
        h1: Our reference model which we would like to be compatible with.
        h2: Our new model which will be the updated model.
        lambda_c: A float between 0.0 and 1.0, which is a regularization
            parameter that determines how much we want to penalize model h2
            for being incompatible with h1. Lower values panalize less and
            higher values penalize more.
    """

    def __init__(self, h1, h2, lambda_c, discriminan_pivot=0.5, **kwargs):
        super(BCBinaryCrossEntropyLoss, self).__init__()
        self.h1 = h1
        self.h2 = h2
        self.loss = F.binary_cross_entropy
        self.lambda_c = lambda_c
        self.discriminan_pivot = discriminan_pivot

    def dissonance(self, h2_support_output_sigmoid, target_labels):
        cross_entropy_loss = F.binary_cross_entropy(h2_support_output_sigmoid, target_labels)
        return cross_entropy_loss

    def forward(self, x, y, reduction="mean"):
        with torch.no_grad():
            h1_output_sigmoid = self.h1(x)

        h1_output_labels = torch.tensor(
            (h1_output_sigmoid >= self.discriminan_pivot),
            dtype=torch.int).view(y.size(0))

        h1_diff = (h1_output_labels - y).float()
        h1_correct = (h1_diff == 0)
        x_support = x[h1_correct]
        y_support = y[h1_correct]
        h2_output_sigmoid = self.h2(x)

        dissonance = 0.0
        if (h1_diff == 0.0).sum(dim=0).item() > 0:
            h2_support_output_sigmoid = self.h2(x_support)
            dissonance = self.dissonance(h2_support_output_sigmoid, y_support)

        base_loss = self.loss(h2_output_sigmoid, y, reduction=reduction)
        new_error_loss = base_loss + self.lambda_c * dissonance
        return new_error_loss


class BCKLDivergenceLoss(nn.Module):
    """
    Backward Compatibility Kullback–Leibler Divergence Loss

    This class implements the backward compatibility loss function
    with the underlying loss function being the Kullback–Leibler
    Divergence loss.

    Example usage:
        h1 = MyModel()
        ... train h1 ...
        h1.eval() (it is important that h1 be put in evaluation mode)

        lambda_c = 0.5 (regularization parameter)
        h2 = MyNewModel() (this may be the same model type as MyModel)
        bcloss = BCKLDivergenceLoss(
            h1, h2, lambda_c, num_classes=num_classes)

        for x, y in training_data:
            loss = bcloss(x, y)
            loss.backward()

        Note that we pass in the input and the target directly to the
        bcloss function instance. It calculates the outputs of h1 and h2
        internally.

    Args:
        h1: Our reference model which we would like to be compatible with.
        h2: Our new model which will be the updated model.
        lambda_c: A float between 0.0 and 1.0, which is a regularization
            parameter that determines how much we want to penalize model h2
            for being incompatible with h1. Lower values panalize less and
            higher values penalize more.
        num_classes: An integer denoting the number of classes that we are
            attempting to classify the input into.
    """

    def __init__(self, h1, h2, lambda_c, num_classes=None, **kwargs):
        super(BCKLDivergenceLoss, self).__init__()
        self.h1 = h1
        self.h2 = h2
        self.loss = F.kl_div
        self.lambda_c = lambda_c
        self.num_classes = num_classes

    def dissonance(self, h2_output_log_softmax, target_labels):
        batch_size = target_labels.size(0)
        kl_div_loss = F.kl_div(
            h2_output_log_softmax,
            utils.labels_to_probabilities(
                target_labels,
                num_classes=self.num_classes,
                batch_size=batch_size))
        return kl_div_loss

    def forward(self, x, y, reduction="batchmean"):
        batch_size = y.size(0)
        with torch.no_grad():
            h1_output_logit, h1_output_softmax, h1_output_logsoftmax = self.h1(x)
        h1_diff = (torch.argmax(h1_output_logsoftmax, 1) - y).float()
        h1_correct = (h1_diff == 0)
        x_support = x[h1_correct]
        y_support = y[h1_correct]
        h2_output_logit, h2_output_softmax, h2_output_logsoftmax = self.h2(x)

        dissonance = 0.0
        if (h1_diff == 0.0).sum(dim=0).item() > 0:
            _, _, h2_support_output_logsoftmax = self.h2(x_support)
            dissonance = self.dissonance(h2_support_output_logsoftmax, y_support)

        base_loss = F.kl_div(
            h2_output_logsoftmax,
            utils.labels_to_probabilities(y, num_classes=self.num_classes, batch_size=batch_size),
            reduction=reduction)
        new_error_loss = base_loss + self.lambda_c * dissonance
        return new_error_loss
