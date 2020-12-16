import tensorflow.compat.v2 as tf


class BCNLLLoss(object):

    def __init__(self, h1, h2, lambda_c):
        self.h1 = h1
        self.h2 = h2
        self.lambda_c = lambda_c
        self.__name__ = "BCNLLLoss"

    def nll_loss(self, target_labels, model_output):
        model_output_labels = tf.argmax(model_output, axis=1)
        model_output_diff = model_output_labels - target_labels
        model_output_correct = (model_output_diff == 0)
        _, model_output_support = tf.dynamic_partition(
            model_output, tf.dtypes.cast(model_output_correct, tf.int32), 2)
        loss = tf.reduce_sum(tf.math.log(model_output_support))

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

        return tf.reduce_sum(new_error_loss)


class BCCrossEntropyLoss(object):

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
