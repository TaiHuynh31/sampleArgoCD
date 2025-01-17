from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import os

def preprocess_data(output_path: str):
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, "X_train.npy"), X_train)
    np.save(os.path.join(output_path, "y_train.npy"), y_train)
    np.save(os.path.join(output_path, "X_test.npy"), X_test)
    np.save(os.path.join(output_path, "y_test.npy"), y_test)

if __name__ == "__main__":
    preprocess_data("/data")
