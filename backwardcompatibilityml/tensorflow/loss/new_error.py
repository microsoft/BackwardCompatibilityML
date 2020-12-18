import tensorflow.compat.v2 as tf


class BCNLLLoss(object):
    """
    Backward Compatibility New Error Negative Log Likelihood Loss

    This class implements the backward compatibility loss function
    with the underlying loss function being the Negative Log Likelihood
    loss.

    Note that the final layer of each model is assumed to have a
    softmax output.

    Example usage:
        h1 = MyModel()
        ... train h1 ...
        h1.trainable = False

        lambda_c = 0.5 (regularization parameter)
        h2 = MyNewModel() (this may be the same model type as MyModel)
        bcloss = BCNLLLoss(h1, h2, lambda_c)
        optimizer = tf.keras.optimizers.SGD(0.01)

        tf_helpers.bc_fit(
            h2,
            training_set=ds_train,
            testing_set=ds_test,
            epochs=6,
            bc_loss=bc_loss,
            optimizer=optimizer)

    Args:
        h1: Our reference model which we would like to be compatible with.
        h2: Our new model which will be the updated model.
        lambda_c: A float between 0.0 and 1.0, which is a regularization
            parameter that determines how much we want to penalize model h2
            for being incompatible with h1. Lower values panalize less and
            higher values penalize more.
    """

    def __init__(self, h1, h2, lambda_c, clip_value_min=1e-10, clip_value_max=4.0):
        self.h1 = h1
        self.h2 = h2
        self.lambda_c = lambda_c
        self.clip_value_min = clip_value_min
        self.clip_value_max = clip_value_max
        self.__name__ = "BCNLLLoss"

    def nll_loss(self, target_labels, model_output):
        # Pick the model output probabilities corresponding to the ground truth labels
        model_outputs_for_targets = tf.gather(
            model_output, tf.dtypes.cast(target_labels, tf.int32), axis=1)
        # We make sure to clip the probability values so that they do not
        # result in Nan's once we take the logarithm
        model_outputs_for_targets = tf.clip_by_value(
            model_outputs_for_targets,
            clip_value_min=self.clip_value_min,
            clip_value_max=self.clip_value_max)
        loss = -1 * tf.reduce_mean(tf.math.log(model_outputs_for_targets))

        return loss

    def dissonance(self, h2_output, target_labels):
        nll_loss = self.nll_loss(target_labels, h2_output)
        return nll_loss

    def __call__(self, x, y):
        h1_output = tf.argmax(self.h1(x), axis=1)
        h2_output = self.h2(x)
        h1_diff = h1_output - y
        h1_correct = (h1_diff == 0)
        _, x_support = tf.dynamic_partition(x, tf.dtypes.cast(h1_correct, tf.int32), 2)
        _, y_support = tf.dynamic_partition(y, tf.dtypes.cast(h1_correct, tf.int32), 2)
        h2_support_output = self.h2(x_support)
        dissonance = self.dissonance(h2_support_output, y_support)
        new_error_loss = self.nll_loss(y, h2_output) + self.lambda_c * dissonance

        return new_error_loss


class BCCrossEntropyLoss(object):
    """
    Backward Compatibility New Error Cross Entropy Loss

    This class implements the backward compatibility loss function
    with the underlying loss function being the Negative Log Likelihood
    loss.

    Note that the final layer of each model is assumed to have a
    softmax output.

    Example usage:
        h1 = MyModel()
        ... train h1 ...
        h1.trainable = False

        lambda_c = 0.5 (regularization parameter)
        h2 = MyNewModel() (this may be the same model type as MyModel)
        bcloss = BCCrossEntropyLoss(h1, h2, lambda_c)
        optimizer = tf.keras.optimizers.SGD(0.01)

        tf_helpers.bc_fit(
            h2,
            training_set=ds_train,
            testing_set=ds_test,
            epochs=6,
            bc_loss=bc_loss,
            optimizer=optimizer)

    Args:
        h1: Our reference model which we would like to be compatible with.
        h2: Our new model which will be the updated model.
        lambda_c: A float between 0.0 and 1.0, which is a regularization
            parameter that determines how much we want to penalize model h2
            for being incompatible with h1. Lower values panalize less and
            higher values penalize more.
    """

    def __init__(self, h1, h2, lambda_c):
        self.h1 = h1
        self.h2 = h2
        self.lambda_c = lambda_c
        self.__name__ = "BCCrossEntropyLoss"
        self.cce_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM)

    def dissonance(self, h2_output, target_labels):
        cross_entropy_loss = self.cce_loss(target_labels, h2_output)
        return cross_entropy_loss

    def __call__(self, x, y):
        h1_output = tf.argmax(self.h1(x), axis=1)
        h2_output = self.h2(x)
        h1_diff = h1_output - y
        h1_correct = (h1_diff == 0)
        _, x_support = tf.dynamic_partition(x, tf.dtypes.cast(h1_correct, tf.int32), 2)
        _, y_support = tf.dynamic_partition(y, tf.dtypes.cast(h1_correct, tf.int32), 2)
        h2_support_output = self.h2(x_support)
        dissonance = self.dissonance(h2_support_output, y_support)
        new_error_loss = self.cce_loss(y, h2_output) + self.lambda_c * dissonance

        return tf.reduce_sum(new_error_loss)


class BCBinaryCrossEntropyLoss(object):
    """
    Backward Compatibility New Error Binary Cross Entropy Loss

    This class implements the backward compatibility loss function
    with the underlying loss function being the Negative Log Likelihood
    loss.

    Note that the final layer of each model is assumed to have a
    softmax output.

    Example usage:
        h1 = MyModel()
        ... train h1 ...
        h1.trainable = False

        lambda_c = 0.5 (regularization parameter)
        h2 = MyNewModel() (this may be the same model type as MyModel)
        bcloss = BCBinaryCrossEntropyLoss(h1, h2, lambda_c)
        optimizer = tf.keras.optimizers.SGD(0.01)

        tf_helpers.bc_fit(
            h2,
            training_set=ds_train,
            testing_set=ds_test,
            epochs=6,
            bc_loss=bc_loss,
            optimizer=optimizer)

    Args:
        h1: Our reference model which we would like to be compatible with.
        h2: Our new model which will be the updated model.
        lambda_c: A float between 0.0 and 1.0, which is a regularization
            parameter that determines how much we want to penalize model h2
            for being incompatible with h1. Lower values panalize less and
            higher values penalize more.
    """

    def __init__(self, h1, h2, lambda_c):
        self.h1 = h1
        self.h2 = h2
        self.lambda_c = lambda_c
        self.__name__ = "BCBinaryCrossEntropyLoss"
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM)

    def dissonance(self, h2_output, target_labels):
        cross_entropy_loss = self.bce_loss(target_labels, h2_output)
        return cross_entropy_loss

    def __call__(self, x, y):
        h1_output = tf.argmax(self.h1(x), axis=1)
        h2_output = self.h2(x)
        h1_diff = h1_output - tf.argmax(y, axis=1)
        h1_correct = (h1_diff == 0)
        _, x_support = tf.dynamic_partition(x, tf.dtypes.cast(h1_correct, tf.int32), 2)
        _, y_support = tf.dynamic_partition(y, tf.dtypes.cast(h1_correct, tf.int32), 2)
        h2_support_output = self.h2(x_support)
        dissonance = self.dissonance(h2_support_output, y_support)
        new_error_loss = self.bce_loss(y, h2_output) + self.lambda_c * dissonance

        return tf.reduce_sum(new_error_loss)


class BCKLDivLoss(object):
    """
    Backward Compatibility New Error Kullback Liebler Divergence Loss

    This class implements the backward compatibility loss function
    with the underlying loss function being the Negative Log Likelihood
    loss.

    Note that the final layer of each model is assumed to have a
    softmax output.

    Example usage:
        h1 = MyModel()
        ... train h1 ...
        h1.trainable = False

        lambda_c = 0.5 (regularization parameter)
        h2 = MyNewModel() (this may be the same model type as MyModel)
        bcloss = BCKLDivLoss(h1, h2, lambda_c)
        optimizer = tf.keras.optimizers.SGD(0.01)

        tf_helpers.bc_fit(
            h2,
            training_set=ds_train,
            testing_set=ds_test,
            epochs=6,
            bc_loss=bc_loss,
            optimizer=optimizer)

    Args:
        h1: Our reference model which we would like to be compatible with.
        h2: Our new model which will be the updated model.
        lambda_c: A float between 0.0 and 1.0, which is a regularization
            parameter that determines how much we want to penalize model h2
            for being incompatible with h1. Lower values panalize less and
            higher values penalize more.
    """

    def __init__(self, h1, h2, lambda_c):
        self.h1 = h1
        self.h2 = h2
        self.lambda_c = lambda_c
        self.__name__ = "BCKLDivLoss"
        self.kldiv_loss = tf.keras.losses.KLDivergence(
            reduction=tf.keras.losses.Reduction.SUM)

    def dissonance(self, h2_output, target_labels):
        kldiv_loss = self.kldiv_loss(target_labels, h2_output)
        return kldiv_loss

    def __call__(self, x, y):
        h1_output = tf.argmax(self.h1(x), axis=1)
        h2_output = self.h2(x)
        h1_diff = h1_output - tf.argmax(y, axis=1)
        h1_correct = (h1_diff == 0)
        _, x_support = tf.dynamic_partition(x, tf.dtypes.cast(h1_correct, tf.int32), 2)
        _, y_support = tf.dynamic_partition(y, tf.dtypes.cast(h1_correct, tf.int32), 2)
        h2_support_output = self.h2(x_support)
        dissonance = self.dissonance(h2_support_output, y_support)
        new_error_loss = self.kldiv_loss(y, h2_output) + self.lambda_c * dissonance

        return tf.reduce_sum(new_error_loss)
