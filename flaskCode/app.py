from flask import Flask, request, jsonify
from flask_cors import CORS  # To allow React to connect
from tensorflow.keras.models import load_model
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your model
model = load_model('model.h5')

# Load tokenizer/vectorizer if needed
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    news_text = data.get('news')

    # Preprocess: tokenize, pad, etc. (this part depends on your model)
    sequence = tokenizer.texts_to_sequences([news_text])
    # Add padding if needed
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    padded = pad_sequences(sequence, maxlen=200)  # Use the same maxlen used during training

    # Predict
    prediction = model.predict(padded)
    result = "Real" if prediction[0][0] >= 0.5 else "Fake"

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(port=8080)
