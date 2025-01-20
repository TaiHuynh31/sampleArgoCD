import tensorflow as tf
import numpy as np
import os
import logging
logging.basicConfig(level=logging.INFO)
logging.info("Starting preprocessing...")

def train_model(dataset_path: str, model_output_path: str):
    X_train_path = os.path.join(dataset_path, "X_train.npy")
    if not os.path.exists(X_train_path):
        raise FileNotFoundError(
            f"Preprocessed data not found at {X_train_path}. Ensure the preprocessing step completed successfully.")
    X_train = np.load(os.path.join(dataset_path, "X_train.npy"))
    y_train = np.load(os.path.join(dataset_path, "y_train.npy"))
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=8)
    os.makedirs(model_output_path, exist_ok=True)
    model.save(os.path.join(model_output_path, "iris_model.h5"))

if __name__ == "__main__":
    train_model("/data", "/model")