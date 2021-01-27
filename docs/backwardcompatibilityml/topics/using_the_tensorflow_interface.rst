.. _using_the_tensorflow_interface:

The Tensorflow Interface
=================================================================

There are two ways in which you may use the notion of the backward
compatibility loss with respect to the Tensorflow framework.

- Creating and training a model subclassed from BCNewErrorCompatibilityModel
- Training a general model h2 using backward compatibility Loss

The first method for achieving this allows you to leverage the existing
``h2.fit(...)`` method in order to train your model. However, it requires
that ``h2`` was instantiated from a model subclassed from BCNewErrorCompatibilityModel.
If you already had a model instantiated from a distinct model class, then
you may need to go through soe effort to extract the layers you need
and wrap them within a new model class subclassed from BCNewErrorCompatibilityModel.

On the other hand, the second method places no restrictions on the architecture
of ``h2``. However, we will be unable to use the existing Tensorflow
``h2.fit(...)`` method to train our model. Instead we will need to train
it using ``tf_helpers.bc_fit``.

We go into the details of both methods below.


Creating and training a model subclassed from BCNewErrorCompatibilityModel
---------------------------------------------------------------------------

Assuming that you have a pre-trained model ``h1`` and that you want to
create a new model ``h2`` that is to be trained using the
backward compatibility loss, with respect to ``h1`` for some value of
``lambda_c``.

With all the parameters as specified above, proceed as follows::

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


An example notebook that walks you through a working example may be found at
``./examples/tensorflow-MNIST`` from the BackwardCompatibilityML project root.


Training a general model h2 using backward compatibility loss
--------------------------------------------------------------

We assume that we have an existing pre-trained model ``h1``.
We instantiate a model ``h2`` as a standard Sequential Keras
model.

With all the parameters as specified above, proceed as follows::

    h2 = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
      tf.keras.layers.Dense(128,activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    lambda_c = 0.9
    h1.trainable = False
    bc_loss = BCCrossEntropyLoss(model, h2, lambda_c)

    optimizer = tf.keras.optimizers.Adam(0.001)

    tf_helpers.bc_fit(
        h2,
        training_set=ds_train,
        testing_set=ds_test,
        epochs=6,
        bc_loss=bc_loss,
        optimizer=optimizer)


An example notebook that walks you through a working example may be found at
``./examples/tensorflow-MNIST-generalized`` from the BackwardCompatibilityML project root.
