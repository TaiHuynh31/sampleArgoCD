from kfp.dsl import  component, Output, Dataset

@component(base_image='python:3.9', packages_to_install=['tensorflow', 'numpy'])
def preprocess_op(output_dataset: Output[Dataset]):
    import tensorflow as tf
    import numpy as np
    import os

    # Load and preprocess CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    #create a dataset dir to store the preprocessed data
    os.makedirs(output_dataset.path, exist_ok=True)

    # Store the preprocessed data
    np.save(os.path.join(output_dataset.path, 'x_train.npy'), x_train)
    np.save(os.path.join(output_dataset.path, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dataset.path, 'x_test.npy'), x_test)
    np.save(os.path.join(output_dataset.path, 'y_test.npy'), y_test)