import os
import logging
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import time

# Force TensorFlow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Paths to model and vectorizer
MODEL_PATH = 'ann_model_improved.keras'
VECTORIZER_PATH = 'vectorizer.pkl'

# Load model and vectorizer
def load_model_and_vectorizer():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Ensure it exists in the project directory.")
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Vectorizer file not found. Ensure it exists in the project directory.")
    
    model = load_model(MODEL_PATH)
    with open(VECTORIZER_PATH, 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        logging.info("Processing request...")

        # Get form inputs
        tags = request.form.get('tags', '').lower()
        category = request.form.get('category', '').lower()
        cuisine_type = request.form.get('cuisine_type', '').lower()
        amenities = request.form.get('amenities', '').lower()
        wheelchair = request.form.get('wheelchair', 'no').lower()
        rating = float(request.form.get('rating', 0))

        # Load and filter dataset
        columns_to_load = ['name', 'category', 'rating', 'reviews_count', 'popularity_score', 
                           'address', 'imageUrls', 'latitude', 'longitude', 'url', 'wheelchair_accessible']
        data = pd.read_csv('mauritiusDataset.csv', usecols=columns_to_load)
        filtered_data = data[data['rating'] >= rating]
        if wheelchair == 'yes':
            filtered_data = filtered_data[filtered_data['wheelchair_accessible'].str.contains('yes', case=False, na=False)]
        if filtered_data.empty:
            return jsonify({'error': 'No recommendations found.'}), 404

        # Vectorize input
        input_text = f"{tags} {category} {cuisine_type} {amenities}"
        input_vec = vectorizer.transform([input_text]).todense()

        # Prepare numerical data
        numerical_data = filtered_data[['rating', 'reviews_count', 'popularity_score']].fillna(0).to_numpy()
        input_vec_repeated = np.repeat(input_vec, numerical_data.shape[0], axis=0)
        combined_input = np.hstack([input_vec_repeated, numerical_data])

        # Predict in batches
        BATCH_SIZE = 100
        predictions = []
        for i in range(0, len(filtered_data), BATCH_SIZE):
            batch = combined_input[i:i + BATCH_SIZE]
            predictions.extend(model.predict(batch).flatten())

        # Add scores and sort
        filtered_data['score'] = predictions
        recommendations = filtered_data.sort_values(by='score', ascending=False).head(10)

        # Return results
        display_columns = ['name', 'category', 'rating', 'address', 'imageUrls', 'latitude', 'longitude', 'url', 'popularity_score']
        recommendations = recommendations[display_columns]
        logging.info(f"Recommendation generation completed in {time.time() - start_time:.2f} seconds.")
        return render_template('index.html', recommendations=recommendations.to_dict(orient='records'))

    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500
    
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
