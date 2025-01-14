from kfp.dsl import component, Input, Dataset, Model

@component(
    base_image='python:3.9',
    packages_to_install=['tensorflow', 'numpy']
)
def evaluate_op(dataset: Input[Dataset], model: Input[Model]):
    import tensorflow as tf
    import numpy as np
    import os

    # Load preprocessed test data
    x_test = np.load(os.path.join(dataset.path, 'x_test.npy'))
    y_test = np.load(os.path.join(dataset.path, 'y_test.npy'))

    # Load the trained model
    model = tf.keras.models.load_model(os.path.join(model.path, 'resnet_model.h5'))

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
