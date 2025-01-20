from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import logging
logging.basicConfig(level=logging.INFO)
logging.info("Starting preprocessing...")
def preprocess_data(output_path: str):
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    os.makedirs(output_path, exist_ok=True)
    pd.DataFrame(X_train).to_csv(os.path.join(output_path, "X_train.csv"), index=False)
    print("Saved X_train.csv to ", output_path)
    pd.DataFrame(y_train).to_csv(os.path.join(output_path, "y_train.csv"), index=False)
    print("Saved y_train.csv to ", output_path)
    pd.DataFrame(X_test).to_csv(os.path.join(output_path, "X_test.csv"), index=False)
    print("Saved X_test.csv to ", output_path)
    pd.DataFrame(y_test).to_csv(os.path.join(output_path, "y_test.csv"), index=False)
    print("Saved y_test.csv to ", output_path)
    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess_data("/data")