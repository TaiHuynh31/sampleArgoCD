from kfp.dsl import component, Input, Dataset, Model

#
# @component(base_image='python:3.9', packages_to_install=['tensorflow', 'numpy', 'scikit-learn'])
# def evaluate_op(dataset: Input[Dataset], model: Input[Model]):
#     import tensorflow as tf
#     import numpy as np
#     from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#     import json
#     import os
#     # Load test data
#     X_test = np.load(os.path.join(dataset.path, 'X_test.npy'))
#     y_test = np.load(os.path.join(dataset.path, 'y_test.npy'))
#
#     # Load the trained model
#     loaded_model = tf.keras.models.load_model(os.path.join(model.path, 'iris_model.h5'))
#
#     # Make predictions
#     y_pred = loaded_model.predict(X_test)
#     y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
#
#     # Compute evaluation metrics
#     accuracy = accuracy_score(y_test, y_pred_classes)
#     precision = precision_score(y_test, y_pred_classes, average='weighted')
#     recall = recall_score(y_test, y_pred_classes, average='weighted')
#     f1 = f1_score(y_test, y_pred_classes, average='weighted')
#
#     # Print metrics to the console
#     print("Evaluation Metrics:")
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1-Score: {f1:.4f}")
#
#     # Save metrics to a JSON file (optional)
#     metrics = {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1_score': f1
#     }
#     os.makedirs(model.path, exist_ok=True)
#     with open(os.path.join(model.path, 'metrics.json'), 'w') as f:
#         json.dump(metrics, f)

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

def evaluate_model(dataset_path: str, model_path: str, output_path: str):
    # Load test data
    X_test = np.load(os.path.join(dataset_path, 'X_test.npy'))
    y_test = np.load(os.path.join(dataset_path, 'y_test.npy'))

    # Load the trained model
    model = tf.keras.models.load_model(os.path.join(model_path, 'iris_model.h5'))

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Compute evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred_classes),
        "precision": precision_score(y_test, y_pred_classes, average='weighted'),
        "recall": recall_score(y_test, y_pred_classes, average='weighted'),
        "f1_score": f1_score(y_test, y_pred_classes, average='weighted')
    }

    # Save metrics to a JSON file
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    # Print metrics to console
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    dataset_path = "/data"
    model_path = "/model"
    output_path = "/metrics"
    evaluate_model(dataset_path, model_path, output_path)
