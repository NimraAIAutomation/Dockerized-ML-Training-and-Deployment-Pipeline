import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="data/dataset.csv"):

    df = pd.read_csv(path)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Dataset loaded:", df.shape)
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])
    print("Features:", X.shape[1])

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    load_data()
