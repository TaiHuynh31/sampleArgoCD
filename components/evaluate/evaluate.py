import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

def evaluate_model(dataset_path: str, model_path: str, metrics_output_path: str):
    # Load data using pandas
    X_test = pd.read_csv(os.path.join(dataset_path, "X_test.csv")).values
    y_test = pd.read_csv(os.path.join(dataset_path, "y_test.csv")).values

    # Load the model
    model = tf.keras.models.load_model(os.path.join(model_path, "iris_model.h5"))

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred_classes),
        "precision": precision_score(y_test, y_pred_classes, average='weighted'),
        "recall": recall_score(y_test, y_pred_classes, average='weighted'),
        "f1_score": f1_score(y_test, y_pred_classes, average='weighted')
    }

    # Save metrics
    os.makedirs(metrics_output_path, exist_ok=True)
    with open(os.path.join(metrics_output_path, "metrics.json"), "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    evaluate_model("/data", "/model", "/metrics")