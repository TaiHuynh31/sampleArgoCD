from kfp.dsl import component, Input, Output, Model, Dataset

@component(base_image='python:3.9', packages_to_install=['tensorflow', 'numpy'])
def train_op(dataset: Input[Dataset], output_model: Output[Model]):
    import tensorflow as tf
    import numpy as np
    import os

    # Load preprocessed data
    x_train = np.load(f'{dataset.path}/x_train.npy')
    y_train = np.load(f'{dataset.path}/y_train.npy')

    # Define and compile the model
    model = tf.keras.applications.ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10)

    # Create directory for saving the model
    os.makedirs(output_model.path, exist_ok=True)

    # Save the trained model
    model.save(os.path.join(output_model.path, 'resnet_model.h5'))

