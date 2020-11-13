# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from backwardcompatibilityml.helpers import utils


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

    def __init__(self, h1, h2, lambda_c, **kwargs):
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
        strict_imitation_loss = base_loss + self.lambda_c * dissonance
        return strict_imitation_loss


class StrictImitationCrossEntropyLoss(nn.Module):
    """
    Strict Imitation Cross-entropy Loss

    This class implements the strict imitation loss function
    with the underlying loss function being the cross-entropy loss.

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

    def __init__(self, h1, h2, lambda_c, **kwargs):
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

        base_loss = self.loss(h2_output_logit, y, reduction=reduction)
        strict_imitation_loss = base_loss + self.lambda_c * dissonance
        return strict_imitation_loss


class StrictImitationBinaryCrossEntropyLoss(nn.Module):
    """
    Strict Imitation Binary Cross-entropy Loss

    This class implements the strict imitation loss function
    with the underlying loss function being the cross-entropy loss.

    Example usage:
        h1 = MyModel()
        ... train h1 ...
        h1.eval() (it is important that h1 be put in evaluation mode)

        lambda_c = 0.5 (regularization parameter)
        h2 = MyNewModel() (this may be the same model type as MyModel)
        siloss = StrictImitationBinaryCrossEntropyLoss(h1, h2, lambda_c)

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

    def __init__(self, h1, h2, lambda_c, discriminant_pivot=0.5, **kwargs):
        super(StrictImitationBinaryCrossEntropyLoss, self).__init__()
        self.h1 = h1
        self.h2 = h2
        self.loss = F.binary_cross_entropy
        self.lambda_c = lambda_c
        self.discriminant_pivot = discriminant_pivot

    def dissonance(self, h1_output_sigmoid, h2_output_sigmoid):
        # Todo: (Xavier) Document that the reason we calculate the
        # dissonance using Negative Log Likelihood is due to the fact
        # that the Pytorch Binary Cross-entropy loss function accepts
        # parameters (input, target) where input is a sigmoid and target
        # is a class label. So that if implemented using Binary Cross-entropy
        # Loss, we would be calculating the same dissonance as
        # the New Error loss.
        nll = torch.sum(
            -1 * h1_output_sigmoid * h2_output_sigmoid.log() + (
                -1 * (1 - h1_output_sigmoid) * (1 - h2_output_sigmoid).log())
        )
        return nll

    def forward(self, x, y, reduction="mean"):
        with torch.no_grad():
            h1_output_sigmoid = self.h1(x)

        h1_output_labels = torch.tensor((h1_output_sigmoid >= self.discriminant_pivot), dtype=torch.int).view(
            y.size(0))
        h1_diff = (h1_output_labels - y).float()
        h1_correct = (h1_diff == 0)
        x_support = x[h1_correct]
        h2_output_sigmoid = self.h2(x)

        dissonance = 0.0
        if (h1_diff == 0.0).sum(dim=0).item() > 0:
            h1_support_output_sigmoid = self.h1(x_support)
            h2_support_output_sigmoid = self.h2(x_support)
            dissonance = self.dissonance(h1_support_output_sigmoid, h2_support_output_sigmoid)

        base_loss = self.loss(h2_output_sigmoid, y, reduction=reduction)
        strict_imitation_loss = base_loss + self.lambda_c * dissonance
        return strict_imitation_loss


class StrictImitationKLDivergenceLoss(nn.Module):
    """
    Strict Imitation Kullback–Leibler Divergence Loss

    This class implements the strict imitation loss function
    with the underlying loss function being the Kullback–Leibler
    Divergence loss.

    Example usage:
        h1 = MyModel()
        ... train h1 ...
        h1.eval() (it is important that h1 be put in evaluation mode)

        lambda_c = 0.5 (regularization parameter)
        h2 = MyNewModel() (this may be the same model type as MyModel)
        siloss = StrictImitationKLDivergenceLoss(
        h1, h2, lambda_c, num_classes=num_classes)

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
        num_classes: An integer denoting the number of classes that we are
            attempting to classify the input into.
    """

    def __init__(self, h1, h2, lambda_c, num_classes=None, **kwargs):
        super(StrictImitationKLDivergenceLoss, self).__init__()
        self.h1 = h1
        self.h2 = h2
        self.loss = F.kl_div
        self.lambda_c = lambda_c
        self.num_classes = num_classes

    def dissonance(self, h1_output_logsoftmax, h2_output_logsoftmax):
        kl_div_loss = F.kl_div(h1_output_logsoftmax, h2_output_logsoftmax)
        return kl_div_loss

    def forward(self, x, y, reduction="batchmean"):
        batch_size = y.size(0)
        with torch.no_grad():
            h1_output_logit, h1_output_softmax, h1_output_logsoftmax = self.h1(x)
        h1_diff = (torch.argmax(h1_output_logsoftmax, 1) - y).float()
        h1_correct = (h1_diff == 0)
        x_support = x[h1_correct]
        h2_output_logit, h2_output_softmax, h2_output_logsoftmax = self.h2(x)

        dissonance = 0.0
        if (h1_diff == 0.0).sum(dim=0).item() > 0:
            _, _, h1_support_output_logsoftmax = self.h1(x_support)
            _, _, h2_support_output_logsoftmax = self.h2(x_support)
            dissonance = self.dissonance(h1_support_output_logsoftmax, h2_support_output_logsoftmax)

        base_loss = self.loss(
            h2_output_logsoftmax,
            utils.labels_to_probabilities(y, num_classes=self.num_classes, batch_size=batch_size),
            reduction=reduction)
        strict_imitation_loss = base_loss + self.lambda_c * dissonance
        return strict_imitation_loss
