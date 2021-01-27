import tensorflow.compat.v2 as tf


def bc_fit(h2, training_set=None, testing_set=None, epochs=None, bc_loss=None, optimizer=None):
    """
    This function is used to train a model h2, using an instance of a Tensorflow BCLoss function
    that has been instantiated using an existing model h1 and regularization parameter lambda_c.

    Example usage:

        h2 = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
          tf.keras.layers.Dense(128,activation='relu'),
          tf.keras.layers.Dense(10, activation='softmax')
        ])

        lambda_c = 0.9
        h1.trainable = False
        bc_loss = BCCrossEntropyLoss(h1, h2, lambda_c)

        optimizer = tf.keras.optimizers.Adam(0.001)

        tf_helpers.bc_fit(
            h2,
            training_set=ds_train,
            testing_set=ds_test,
            epochs=6,
            bc_loss=bc_loss,
            optimizer=optimizer)

    Args:
        h2: A Tensorflow model that we want to train using backward compatibility.
        training_set: The training set for our model.
        testing_set: The testing set for validating our model.
        epochs: The number of training epochs.
        bc_loss: An instance of a Tensorflow BCLoss function.
        optimizer: The optimizer to use.

    Returns:
        Does not return anything. But it updates the weights of the model h2.
    """

    @tf.function
    def compatibility_train_step(x_batch_train, y_batch_train, bc_loss, optimizer, h2):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            # Compute the loss value for this minibatch.
            loss_value = bc_loss(x_batch_train, y_batch_train)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, h2.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, h2.trainable_weights))

        return loss_value

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(training_set):

            loss_value = compatibility_train_step(
                x_batch_train, y_batch_train, bc_loss, optimizer, h2)

            # Log every 10 batches.
            if step % 10 == 0:
                print("=", end="")

        print(
            " Training loss: %.4f"
            % (float(loss_value),)
        )

    print("Training done.")
