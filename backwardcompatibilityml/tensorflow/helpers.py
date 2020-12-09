import tensorflow.compat.v2 as tf


def bc_fit(h2, training_set=None, testing_set=None, epochs=None, bc_loss=None, optimizer=None):
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
