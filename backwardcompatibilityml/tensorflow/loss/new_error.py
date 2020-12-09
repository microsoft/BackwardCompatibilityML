import tensorflow.compat.v2 as tf


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
