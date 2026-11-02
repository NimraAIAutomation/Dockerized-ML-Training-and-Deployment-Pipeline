from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from training.data_loader import load_data
from training.preprocessing import create_preprocessor


def train_models():

    X_train, X_test, y_train, y_test = load_data()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    }

    trained_models = {}

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessing", create_preprocessor()),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)

        trained_models[name] = pipeline

        print(f"{name} trained successfully!")

    return trained_models, X_test, y_test


if __name__ == "__main__":
    train_models()
