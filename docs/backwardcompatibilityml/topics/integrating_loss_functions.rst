.. _integrating_loss_functions:

Integrating the Backward Compatibility ML Loss Functions
========================================================

We have implemented the following compatibility loss functions:

1. ``BCNLLLoss`` - Backward Compatibility Negative Log Likelihood Loss
2. ``BCCrossEntropyLoss``- Backward Compatibility Cross-entropy Loss
3. ``BCBinaryCrossEntropyLoss`` - Backward Compatibility Binary Cross-entropy Loss
4. ``BCKLDivergenceLoss`` - Backward Compatibility Kullback–Leibler Divergence Loss

And the following strict imitation loss functions:

1. ``StrictImitationNLLLoss`` - Strict Imitation Negative Log Likelihood Loss
2. ``StrictImitationCrossEntropyLoss`` - Strict Imitation Cross-entropy Loss
3. ``StrictImitationBinaryCrossEntropyLoss`` - Strict Imitation Binary Cross-entropy Loss
4. ``StrictImitationKLDivergenceLoss`` - Strict Imitation Kullback–Leibler Divergence Loss

Both these sets of loss functions are implemented along the lines of

``compatibility_loss(x, y) = underlying_loss(h2(x), y) + lambda_c * dissonance(h1, h2, x, y)``

Where the dissonance is the backward compatibility dissonance for the compatibility
loss functions, and the strict imitation dissonance in the case of the strict imitation
loss functions.

Example Usage
--------------

Let us assume that we have a pre-trained model ``h1`` that we want to use
as our reference model while training / updating a new model ``h2``.

Let us load our pre-trained model::

    h1 = MyModel()
    h1.load_state_dict(torch.load("path/to/state/dict.state"))

Then let us instantiate ``h2`` and train / update it, using ``h1`` as a
reference::

    from backwardcompatibilityml.loss import BCCrossEntropyLoss

    h2 = MyModel()
    lambda_c = 0.7
    bc_loss = BCCrossEntropyLoss(h1, h2, lambda_c)

    for data, target in updated_training_set:
        h2.zero_grad()
        loss = bc_loss(data, target)
        loss.backward()

Calling ``loss.backward()`` at each step of the training iteration, updates
the weights of the model ``h2``.

You may also decide to use an optimizer as follows::

    import torch.optim as optim
    from backwardcompatibilityml.loss import BCCrossEntropyLoss

    h2 = MyModel()
    lambda_c = 0.7
    learning_rate = 0.01
    momentum = 0.5
    bc_loss = BCCrossEntropyLoss(h1, h2, lambda_c)
    optimizer = optim.SGD(h2.parameters(), lr=learning_rate, momentum=momentum)

    for data, target in updated_training_set:
        loss = bc_loss(data, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

The usage for ``BCNLLLoss``, ``StrictImitationCrossEntropyLoss`` and ``StrictImitationNLLLoss``
is exactly the same as above.

Assumptions on the implementation of h1 and h2
-----------------------------------------------

It is important*to emphasize that since the compatibility and strict imitation loss functions
need to use ``h1`` and ``h2`` to calculate the loss, some assumptions had to be made on the
output returned by ``h1`` and ``h2``.

Specifically, we require that both the models ``h1`` and ``h2`` return an ordered triple
containing:

1. The raw logits output from the final layer.
2. The function softmax applied to the raw logits.
3. The function log_softmax applied to the raw logits.

Here is an example Logistic Regression model satisfying these requirements::

    import torch.nn as nn
    import torch.nn.functional as F


    class LogisticRegression(nn.Module):

        def __init__(self, input_dim, output_dim):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            out = self.linear(x)
            out_softmax = F.softmax(out, dim=-1)
            out_log_softmax = F.log_softmax(out, dim=-1)

            return out, out_softmax, out_log_softmax

Here is an example Convolutional Network model satisfying these requirements::

    import torch.nn as nn
    import torch.nn.functional as F

    class ConvolutionalNetwork(nn.Module):
        def __init__(self):
            super(ConvolutionalNetwork, self).__init__()
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
