import joblib   # type: ignore
import numpy as np  # type: ignore


def predict(model_path, input_data):
    """
    Load the trained model and make predictions on new data.
    """
    # Load the model
    model = joblib.load(model_path)

    # Make predictions
    predictions = model.predict(input_data)
    return predictions


if __name__ == "__main__":
    # Example input data (replace with your own data)
    input_data = np.array([[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]])

    # Make predictions
    predictions = predict("iris_model.pkl", input_data)
    print("Predictions:", predictions)
