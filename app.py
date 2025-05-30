import json
import pickle
from flask import Flask, request, jsonify, app, url_for, render_template

import numpy as np
import pandas as pd

app=Flask(__name__)
regmodel = pickle.load(open('bangalore_home_prices_model.pickle', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    # Extract location names from your model or JSON
    with open("bhp_columns.json") as f:
        data_columns = json.load(f)['data_columns']
    location_names = [k for k in data_columns if k not in ['total_sqft', 'bath', 'bhk']]
    return render_template('home.html', locations=location_names)


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    
    # Ensure input order matches data_columns
    with open("bhp_columns.json") as f:
        data_columns = json.load(f)['data_columns']

    input_data = [0] * len(data_columns)
    
    # Fill in numeric values
    input_data[0] = data['total_sqft']
    input_data[1] = data['bath']
    input_data[2] = data['bhk']

    # Fill in location one-hot encoding
    for loc in data:
        if loc in data_columns:
            idx = data_columns.index(loc)
            input_data[idx] = data[loc]

    prediction = regmodel.predict([input_data])[0]
    return jsonify(prediction)



@app.route('/predict', methods=['POST'])
def predict():
    total_sqft = float(request.form['total_sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])
    location = request.form['location'].lower()

    # Load columns
    with open("bhp_columns.json") as f:
        data_columns = json.load(f)['data_columns']
    location_cols = [key for key in data_columns if key not in ['total_sqft', 'bath', 'bhk']]

    # Create input vector
    input_data = [0] * len(data_columns)
    input_data[0] = total_sqft
    input_data[1] = bath
    input_data[2] = bhk
    if location in location_cols:
        loc_index = list(data_columns).index(location)
        input_data[loc_index] = 1

    # Predict
    prediction = regmodel.predict([input_data])[0]
    return render_template('home.html', prediction_text=f"Predicted House Price is: ₹ {round(prediction, 2)} Lakhs", locations=location_cols)






if __name__ == "__main__":
    app.run(debug=True)
