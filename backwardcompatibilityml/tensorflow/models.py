# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tensorflow.compat.v2 as tf


class BCNewErrorCompatibilityModel(tf.keras.models.Sequential):
    """
    BackwardCompatibility base model for Tensorflow

    You may create a new Tensorflow model by subclassing
    your new model h2 from this model.
    This allows you to train or update a new model h2, using the
    backward compatibility loss, with respect to an existing
    model h1, using the Tensorflow fit method, h2.fit(...).

    Assuming that you have a pre-trained model h1 and you
    would like to create a new model h2 trained
    using the backward compatibility loss with respect to h1,
    the following describes the example usage:

        h1.trainable = False

        h2 = BCNewErrorCompatibilityModel([
          tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
          tf.keras.layers.Dense(128,activation='relu'),
          tf.keras.layers.Dense(10, activation='softmax')
        ], h1=h1, lambda_c=0.7)

        h2.compile(
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=['accuracy']
        )

        h2.fit(
            dataset_train,
            epochs=6,
            validation_data=dataset_test,
        )
    """

    def __init__(self, *args, h1=None, lambda_c=0.0, **kwargs):
        """
        Args:
          h1: An existing tensorflow model that has been pre-trained,
            and which we want to be compatible with.
        lambda_c: A floating point value between 0.0 and 1.0 that is
          used as a regularization parameter to weight how much the
          dissonance is used to penalize the loss while training.
        """
        super(BCNewErrorCompatibilityModel, self).__init__(*args)

        if h1 is None:
            raise Exception("The parameter h1 is required.")
        self.h1 = h1
        self.lambda_c = lambda_c

    def dissonance(self, h2_output, target_labels, loss):
        """
        The dissonance function, which uses the loss function
        specified by the user to calculate the loss on a subset
        of the target.
        """
        calculated_loss = loss(target_labels, h2_output)
        return calculated_loss

    def loss_func(self, x, y, loss=None):
        """
        Backward compatibility loss function to be used by the model
        """
        if loss is None:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.SUM)
        h1_output = tf.argmax(self.h1(x), axis=1)
        h2_output = self(x)
        h1_diff = h1_output - y

        # Here we determine which datapoints were correctly labeled by h1
        h1_correct = (h1_diff == 0)

        # Here we pull those datapoints which were correctly labeled by h1
        _, x_support = tf.dynamic_partition(x, tf.dtypes.cast(h1_correct, tf.int32), 2)

        # Here we pull the ground truth labels for those datapoints which were
        # correctly labeled by h1.
        _, y_support = tf.dynamic_partition(y, tf.dtypes.cast(h1_correct, tf.int32), 2)

        # Here we pull those outputs of h2, on datapoints which were correctly labeled
        # by h1. And use these to calculate the new error dissonance.
        _, h2_support_output = tf.dynamic_partition(h2_output, tf.dtypes.cast(h1_correct, tf.int32), 2)
        new_error_dissonance = self.dissonance(h2_support_output, y_support, loss)

        # We calculate the new error loss.
        new_error_loss = loss(y, h2_output) + self.lambda_c * new_error_dissonance

        return tf.reduce_sum(new_error_loss)

    def train_step(self, data):
        """
        This is a custom train step which allows us to use to train our model
        using the `fit()` method, using a non-standard loss funtion.
        """
        x, y = data

        with tf.GradientTape() as tape:
            # Here we compute the loss using the loss function specified
            # by the user in `compile()`, within the context of our compatibility
            # loss function.
            loss = self.loss_func(x, y, loss=self.compiled_loss)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        y_pred = self(x, training=False)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
