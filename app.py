from flask import Flask, request, jsonify, render_template   # type: ignore
import joblib   # type: ignore
import numpy as np  # type: ignore

app = Flask(__name__)

# Load the trained model
model = joblib.load("iris_model.pkl")

# Map numeric predictions to class labels
class_labels = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}


@app.route("/")
def home():
    """
    Home route to render the index.html template.
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to make predictions using the trained model.
    """
    try:
        # Get input data from the form
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])
        # Prepare input data as a 2D array
        input_data = np.array([
            [sepal_length, sepal_width, petal_length, petal_width]
            ]
            )
        # Make predictions
        prediction = model.predict(input_data)[0]  # Get the first prediction
        predicted_class = class_labels.get(prediction, "Unknown")
        # Return prediction as JSON
        return jsonify({"prediction": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
