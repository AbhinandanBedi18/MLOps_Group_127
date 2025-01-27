import joblib   # type: ignore
import numpy as np  # type: ignore
from sklearn.datasets import load_iris  # type: ignore
from sklearn.model_selection import train_test_split    # type: ignore
from sklearn.metrics import (   # type: ignore
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def test_predict():
    """
    Test the predict function.
    """
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load the trained model
    model = joblib.load("iris_model.pkl")
    assert model is not None, "Model should be loaded"

    # Test prediction on a single sample
    sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example input (2D array)
    prediction = model.predict(sample_input)

    # Check if the prediction is a NumPy integer or Python integer
    assert isinstance(
        prediction[0], (int, np.integer)
    ), "Prediction should be an integer"
    assert 0 <= prediction[0] <= 2, "Prediction of valid class (0, 1, or 2)"

    # Test prediction on the entire test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    assert accuracy < 1.0, "Accuracy of a simpler model less than 1"


if __name__ == "__main__":
    test_predict()
    print("All tests passed!")
