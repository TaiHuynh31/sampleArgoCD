import os
import pandas as pd
import tensorflow as tf

def train_model(dataset_path: str, model_output_path: str):
    X_train = pd.read_csv(os.path.join(dataset_path, "X_train.csv")).values
    y_train = pd.read_csv(os.path.join(dataset_path, "y_train.csv")).values

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