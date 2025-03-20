import os
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = None
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is loaded
        if model is None:
            return render_template('index.html', prediction="❌ Model not loaded. Check server logs.")

        # Get input values
        cgpa = float(request.form['cgpa'])
        iq = float(request.form['iq'])

        print(f"📥 Received Input: CGPA={cgpa}, IQ={iq}")  # Debugging line
        
        # Prepare input array
        input_data = np.array([[cgpa, iq]])

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "✅ Placement Ho Jayega" if prediction == 1 else "❌ Placement Nahi Hoga"

        print(f"🔮 Prediction: {result}")  # Debugging line
        
        return render_template('index.html', prediction=result)

    except Exception as e:
        print(f"❌ Error: {e}")  # Debugging line
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Railway often assigns 8080
    app.run(host="0.0.0.0", port=port, debug=True)
