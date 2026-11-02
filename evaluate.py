
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from training.train import train_models


def evaluate_models():

    trained_models, X_test, y_test = train_models()

    results = {}

    for name, model in trained_models.items():

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities)
        }

        results[name] = metrics

        print(f"\n{name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    best_model = max(
        results,
        key=lambda name: results[name]["recall"]
    )

    print("\nBest model:", best_model)
    print("Selection metric: Recall")

    return trained_models, results, best_model


if __name__ == "__main__":
    evaluate_models()
