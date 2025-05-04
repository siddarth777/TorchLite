import pandas as pd
import numpy as np

def CSVdataloader(path, test_ratio=0.2, shuffle=True, seed=42):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values

    if shuffle:
        np.random.seed(seed)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test

def batch_loader(X, y, batch_size=16):
    indices = np.arange(len(X))
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]

def z_normalize(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
        std[std == 0] = 1e-8  # prevent division by zero

    X_norm = (X - mean) / std
    return X_norm, mean, std