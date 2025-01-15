from kfp.dsl import component, Output, Dataset


@component(base_image='python:3.9', packages_to_install=['scikit-learn', 'numpy'])
def preprocess_op(output_dataset: Output[Dataset]):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    import numpy as np
    import os
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a directory to store the preprocessed data
    os.makedirs(output_dataset.path, exist_ok=True)

    # Save the preprocessed data
    np.save(os.path.join(output_dataset.path, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dataset.path, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dataset.path, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dataset.path, 'y_test.npy'), y_test)