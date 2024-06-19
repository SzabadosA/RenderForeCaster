from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({"predicted_rendertime": prediction[0]})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
