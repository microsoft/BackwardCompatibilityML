# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F


class StrictImitationNLLLoss(nn.Module):
    """
    Strict Imitation Negative Log Likelihood Loss

    This class implements the strict imitation loss function
    with the underlying loss function being the Negative Log Likelihood
    loss.

    Example usage:
        h1 = MyModel()
        ... train h1 ...
        h1.eval() (it is important that h1 be put in evaluation mode)

        lambda_c = 0.5 (regularization parameter)
        h2 = MyNewModel() (this may be the same model type as MyModel)
        siloss = StrictImitationNLLLoss(h1, h2, lambda_c)

        for x, y in training_data:
            loss = siloss(x, y)
            loss.backward()

        Note that we pass in the input and the target directly to the
        siloss function instance. It calculates the outputs of h1 and h2
        internally.

    Args:
        h1: Our reference model which we would like to be compatible with.
        h2: Our new model which will be the updated model.
        lambda_c: A float between 0.0 and 1.0, which is a regularization
            parameter that determines how much we want to penalize model h2
            for being incompatible with h1. Lower values panalize less and
            higher values penalize more.
    """

    def __init__(self, h1, h2, lambda_c):
        super(StrictImitationNLLLoss, self).__init__()
        self.h1 = h1
        self.h2 = h2
        self.loss = F.nll_loss
        self.lambda_c = lambda_c

    def dissonance(self, h1_output_prob, h2_output_prob):
        nll = torch.sum(
            torch.sum(
                -1 * h1_output_prob * h2_output_prob.log(),
                1)
        )
        return nll

    def forward(self, x, y, reduction="mean"):
        with torch.no_grad():
            h1_output_logit, h1_output_softmax, h1_output_logsoftmax = self.h1(x)
        h1_diff = (h1_output_logsoftmax.data.max(1)[1] - y).float()
        h1_correct = (h1_diff == 0)
        x_support = x[h1_correct]
        h2_output_logit, h2_output_softmax, h2_output_logsoftmax = self.h2(x)

        dissonance = 0.0
        if (h1_diff == 0.0).sum(dim=0).item() > 0:
            _, h1_support_output_softmax, _ = self.h1(x_support)
            _, h2_support_output_softmax, _ = self.h2(x_support)
            dissonance = self.dissonance(h1_support_output_softmax, h2_support_output_softmax)

        base_loss = self.loss(h2_output_logsoftmax, y, reduction=reduction)
        new_error_loss = base_loss + self.lambda_c * dissonance
        return new_error_loss


class StrictImitationCrossEntropyLoss(nn.Module):
    """
    Strict Imitation Cross Entropy Loss

    This class implements the strict imitation loss function
    with the underlying loss function being the Cross Entropy loss.

    Example usage:
        h1 = MyModel()
        ... train h1 ...
        h1.eval() (it is important that h1 be put in evaluation mode)

        lambda_c = 0.5 (regularization parameter)
        h2 = MyNewModel() (this may be the same model type as MyModel)
        siloss = StrictImitationCrossEntropyLoss(h1, h2, lambda_c)

        for x, y in training_data:
            loss = siloss(x, y)
            loss.backward()

        Note that we pass in the input and the target directly to the
        siloss function instance. It calculates the outputs of h1 and h2
        internally.

    Args:
        h1: Our reference model which we would like to be compatible with.
        h2: Our new model which will be the updated model.
        lambda_c: A float between 0.0 and 1.0, which is a regularization
            parameter that determines how much we want to penalize model h2
            for being incompatible with h1. Lower values panalize less and
            higher values penalize more.
    """

    def __init__(self, h1, h2, lambda_c):
        super(StrictImitationCrossEntropyLoss, self).__init__()
        self.h1 = h1
        self.h2 = h2
        self.loss = F.cross_entropy
        self.lambda_c = lambda_c

    def dissonance(self, h1_output_labels, h2_output_logit):
        cross_entropy_loss = F.cross_entropy(h2_output_logit, h1_output_labels)
        return cross_entropy_loss

    def forward(self, x, y, reduction="mean"):
        with torch.no_grad():
            h1_output_logit, h1_output_softmax, h1_output_logsoftmax = self.h1(x)
        h1_diff = (torch.argmax(h1_output_logsoftmax, 1) - y).float()
        h1_correct = (h1_diff == 0)
        x_support = x[h1_correct]
        h2_output_logit, h2_output_softmax, h2_output_logsoftmax = self.h2(x)

        dissonance = 0.0
        if (h1_diff == 0.0).sum(dim=0).item() > 0:
            h1_support_output_logit, _, _ = self.h1(x_support)
            h1_support_output_labels = torch.argmax(h1_support_output_logit, 1)
            h2_support_output_logit, _, _ = self.h2(x_support)
            dissonance = self.dissonance(h1_support_output_labels, h2_support_output_logit)

        base_loss = self.loss(h2_output_logit, y)
        base_loss = self.loss(h2_output_logit, y, reduction=reduction)
        new_error_loss = base_loss + self.lambda_c * dissonance
        return new_error_loss
