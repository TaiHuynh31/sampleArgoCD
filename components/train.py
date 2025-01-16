from kfp.dsl import component, Input, Output, Model,Dataset


@component(base_image='python:3.9', packages_to_install=['tensorflow', 'numpy'])
def train_op(dataset: Input[Dataset], output_model: Output[Model]):
    import tensorflow as tf
    import numpy as np
    import os
    # Load preprocessed data
    X_train = np.load(os.path.join(dataset.path, 'X_train.npy'))
    y_train = np.load(os.path.join(dataset.path, 'y_train.npy'))

    # Define a simple neural network model

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes for Iris dataset
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=8)

    # Create directory for saving the model
    os.makedirs(output_model.path, exist_ok=True)

    # Save the trained model
    model.save(os.path.join(output_model.path, 'iris_model.h5'))
    # demo somethings