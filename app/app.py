# app.py

import pickle
import bz2file as bz2
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained models
with open('../bin/price_predictor.pkl', 'rb') as f:
    price_model = pickle.load(f)

with open('../bin/label_binarizer.pkl', 'rb') as f:
    label_binarizer = pickle.load(f)

with open('../bin/amenities_predictor.pkl', 'rb') as f:
    amenities_model = pickle.load(f)

# Load categorical data
with open('../bin/cat_data.json', 'r') as f:
    cat_data = json.load(f)

# Routes
@app.route('/')
def home():
    return render_template('index.html', data=cat_data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    processed_data = {}
    for key in ['accommodates', 'bedrooms', 'beds', 'bathrooms']:
        processed_data[key] = [float(data[key])]
    for key in ['cleaning_fee', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']:
        processed_data[key] = [1 if (data[key] and data[key] == 'True') else 0]
    for key in ['property_type', 'room_type', 'cancellation_policy', 'city']:
        processed_data[key] = [data[key]]

    input_data = pd.DataFrame(processed_data)
    price = price_model.predict(input_data)
    price = round(np.exp(price[0]), 2)


    processed_data['price'] = [price]
    input_data = pd.DataFrame(processed_data)
    amenities = amenities_model.predict(input_data)
    amenities_labels = label_binarizer.inverse_transform(amenities)

    return render_template('prediction.html', prediction={'price': price, 'amenities': amenities_labels[0]})

if __name__ == '__main__':
    app.run(debug=True)
