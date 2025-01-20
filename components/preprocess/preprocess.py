from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def preprocess_data(output_path: str):
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    os.makedirs(output_path, exist_ok=True)
    pd.DataFrame(X_train).to_csv(os.path.join(output_path, "X_train.csv"), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(output_path, "y_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(output_path, "X_test.csv"), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(output_path, "y_test.csv"), index=False)

if __name__ == "__main__":
    preprocess_data("/data")