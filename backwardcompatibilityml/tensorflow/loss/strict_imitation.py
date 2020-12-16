import tensorflow.compat.v2 as tf
import tensorflow.compat.v1 as tf1


class BCStrictImitationNLLLoss(object):

    def __init__(self, h1, h2, lambda_c):
        self.h1 = h1
        self.h2 = h2
        self.lambda_c = lambda_c
        self.__name__ = "BCStrictImitationNLLLoss"

    def nll_loss(self, target_labels, model_output):
        model_output_labels = tf.argmax(model_output, axis=1)
        model_output_diff = model_output_labels - target_labels
        model_output_correct = (model_output_diff == 0)
        _, model_output_support = tf.dynamic_partition(
            model_output, tf.dtypes.cast(model_output_correct, tf.int32), 2)
        loss = tf.reduce_sum(tf.math.log(model_output_support))

        return loss

    def dissonance(self, h2_output, target_labels):
        log_loss = tf1.losses.log_loss(target_labels, h2_output, epsilon=1e-07)
        return log_loss

    def __call__(self, x, y):
        h1_output = tf.argmax(self.h1(x), axis=1)
        h2_output = self.h2(x)
        h1_diff = h1_output - tf.argmax(y, axis=1)
        h1_correct = (h1_diff == 0)
        _, x_support = tf.dynamic_partition(x, tf.dtypes.cast(h1_correct, tf.int32), 2)
        _, y_support = tf.dynamic_partition(y, tf.dtypes.cast(h1_correct, tf.int32), 2)
        h2_support_output = self.h2(x_support)
        strict_imitation_dissonance = self.dissonance(h2_support_output, y_support)
        strict_imitation_loss = self.nll_loss(tf.argmax(y, axis=1), h2_output) + self.lambda_c * strict_imitation_dissonance

        return tf.reduce_sum(strict_imitation_loss)


class BCStrictImitationCrossEntropyLoss(object):

    def __init__(self, h1, h2, lambda_c):
        self.h1 = h1
        self.h2 = h2
        self.lambda_c = lambda_c
        self.__name__ = "BCStrictImitationCrossEntropyLoss"
        self.cce_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM)

    def dissonance(self, h2_output, target_labels):
        log_loss = tf1.losses.log_loss(target_labels, h2_output, epsilon=1e-07)
        return log_loss

    def __call__(self, x, y):
        h1_output = tf.argmax(self.h1(x), axis=1)
        h2_output = self.h2(x)
        h1_diff = h1_output - tf.argmax(y, axis=1)
        h1_correct = (h1_diff == 0)
        _, x_support = tf.dynamic_partition(x, tf.dtypes.cast(h1_correct, tf.int32), 2)
        _, y_support = tf.dynamic_partition(y, tf.dtypes.cast(h1_correct, tf.int32), 2)
        h2_support_output = self.h2(x_support)
        strict_imitation_dissonance = self.dissonance(h2_support_output, y_support)
        strict_imitation_loss = self.cce_loss(tf.argmax(y, axis=1), h2_output) + self.lambda_c * strict_imitation_dissonance

        return tf.reduce_sum(strict_imitation_loss)


class BCStrictImitationBinaryCrossEntropyLoss(object):

    def __init__(self, h1, h2, lambda_c):
        self.h1 = h1
        self.h2 = h2
        self.lambda_c = lambda_c
        self.__name__ = "BCStrictImitationBinaryCrossEntropyLoss"
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM)

    def dissonance(self, h2_output, target_labels):
        log_loss = tf1.losses.log_loss(target_labels, h2_output, epsilon=1e-07)
        return log_loss

    def __call__(self, x, y):
        h1_output = tf.argmax(self.h1(x), axis=1)
        h2_output = self.h2(x)
        h1_diff = h1_output - tf.argmax(y, axis=1)
        h1_correct = (h1_diff == 0)
        _, x_support = tf.dynamic_partition(x, tf.dtypes.cast(h1_correct, tf.int32), 2)
        _, y_support = tf.dynamic_partition(y, tf.dtypes.cast(h1_correct, tf.int32), 2)
        h2_support_output = self.h2(x_support)
        strict_imitation_dissonance = self.dissonance(h2_support_output, y_support)
        strict_imitation_loss = self.bce_loss(y, h2_output) + self.lambda_c * strict_imitation_dissonance

        return tf.reduce_sum(strict_imitation_loss)


class BCStrictImitationKLDivLoss(object):

    def __init__(self, h1, h2, lambda_c):
        self.h1 = h1
        self.h2 = h2
        self.lambda_c = lambda_c
        self.__name__ = "BCStrictImitationKLDivLoss"
        self.kldiv_loss = tf.keras.losses.KLDivergence(
            reduction=tf.keras.losses.Reduction.SUM)

    def dissonance(self, h2_output, target_labels):
        log_loss = tf1.losses.log_loss(target_labels, h2_output, epsilon=1e-07)
        return log_loss

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
