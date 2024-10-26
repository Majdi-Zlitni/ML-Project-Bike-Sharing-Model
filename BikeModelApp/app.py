from flask import Flask, request, render_template, redirect, url_for, flash
import joblib
import os
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set this to a random, unique, and secret value

# Load the model
model_path = os.path.join(os.path.dirname(__file__), './../Model/modele_location_velos.pkl')
try:
    model = joblib.load(model_path)
    if not hasattr(model, 'predict'):
        raise ValueError("Loaded object is not a valid model.")
except (FileNotFoundError, ValueError) as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash('Model not loaded. Please check the server logs for more details.')
        return redirect(url_for('home'))
    
    try:
        # Get form data
        features = [
            'instant', 'yr', 'mnth', 'weekday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
            'casual', 'registered', 'income_level', 'user_satisfaction', 'bike_infrastructure_score',
            'air_pollution_index', 'public_transport_access', 'user_activity_level_encoded', 'user_type_encoded',
            'zone_type_encoded', 'temp_hum_interaction', 'casual_registered_ratio', 'high_temp',
            'user_experience_score', 'avg_rentals'
        ]
        
        user_input = [float(request.form[feature]) for feature in features]
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([user_input], columns=features)
        
        # Make prediction
        prediction = model.predict(input_df)
        
        return render_template('index.html', prediction_text=f'Predicted Outcome: {prediction[0]}')
    except Exception as e:
        flash(f'An error occurred: {str(e)}')
        print(f'An error occurred: {str(e)}')
        return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)