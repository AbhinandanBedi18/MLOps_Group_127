from sklearn.datasets import load_iris  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import joblib  # type: ignore
import mlflow  # type: ignore
import mlflow.sklearn  # type: ignore


def train_model(model_path):
    """
    Train a Decision Tree Classifier on the Iris dataset and save the model.
    Log metrics, parameters, and the model using MLflow.
    """
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Start an MLflow run
    with mlflow.start_run():
        # Log a custom tag for the user
        mlflow.set_tag("user", "Pooja")
        # Log parameters
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("max_depth", 2)
        mlflow.log_param("max_features", 2)

        # Train a Decision Tree Classifier
        model = RandomForestClassifier(
            max_depth=2, max_features=2, random_state=42
            )
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)

        # Save the model
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)  # Log the model as an artifact

        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_model("iris_model.pkl")
