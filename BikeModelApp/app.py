from flask import Flask, request, render_template, redirect, url_for, flash, session
import pandas as pd
import joblib
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Load models
try:
    models = {
        'modele_KNN_DSO2': joblib.load('../Model/modele_KNN_DSO2.pkl'),
        'modele_SVM_DSO1': joblib.load('../Model/modele_SVM_DSO1.pkl'),
        'modele_Régression_Linéaire_DSO3': joblib.load('../Model/modele_Régression_Linéaire_DSO3.pkl')
    }
except Exception as e:
    print(f"Error loading models: {e}")
    models = {}

# Verify models are loaded correctly
for name, model in models.items():
    print(f"Loaded {name}: {type(model)}")

# Expected features FOR DSO3
'''
features = [
'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
        'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)',
        'Snowfall (cm)', 'Holiday', 'Functioning Day', 'Seasons_Spring',
        'Seasons_Summer', 'Seasons_Winter'
]
default_values = {    
 'Temperature(°C)': 20.0,
        'Humidity(%)': 60,
        'Wind speed (m/s)': 5.0,
        'Visibility (10m)': 10.0,
        'Dew point temperature(°C)': 10.0,
        'Solar Radiation (MJ/m2)': 100.0,
        'Rainfall(mm)': 0.0,
        'Snowfall (cm)': 0.0,
        'Holiday': 0,
        'Functioning Day': 1,
        'Seasons_Spring': 0,
        'Seasons_Summer': 1,
        'Seasons_Winter': 0
}
'''
features = [
    'Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Dew point temperature(°C)',
    'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)', 'Holiday', 'Functioning Day',
    'Température ressentie', 'Météo extrême', 'Visibilité_catégorisée', 'Indice_confort',
    'Intensité du vent', 'Seasons_Spring', 'Seasons_Summer', 'Seasons_Winter',
    'Saison_précipitations_Autumn_1', 'Saison_précipitations_Spring_0', 'Saison_précipitations_Spring_1',
    'Saison_précipitations_Summer_0', 'Saison_précipitations_Summer_1', 'Saison_précipitations_Winter_0',
    'Saison_précipitations_Winter_1', 'Rented Bike Count'
]
# Default values
default_values = {
    'Hour': 12,
    'Temperature(°C)': 20.0,
    'Humidity(%)': 60,
    'Wind speed (m/s)': 5.0,
    'Dew point temperature(°C)': 10.0,
    'Solar Radiation (MJ/m2)': 100.0,
    'Rainfall(mm)': 0.0,
    'Snowfall (cm)': 0.0,
    'Holiday': 0,
    'Functioning Day': 1,
    'Température ressentie': 20.0,
    'Météo extrême': 0,
    'Visibilité_catégorisée': 1,
    'Indice_confort': 50,
    'Intensité du vent': 10,
    'Seasons_Spring': 0,
    'Seasons_Summer': 1,
    'Seasons_Winter': 0,
    'Saison_précipitations_Autumn_1': 0,
    'Saison_précipitations_Spring_0': 1,
    'Saison_précipitations_Spring_1': 0,
    'Saison_précipitations_Summer_0': 0,
    'Saison_précipitations_Summer_1': 1,
    'Saison_précipitations_Winter_0': 0,
    'Saison_précipitations_Winter_1': 1,
    'Rented Bike Count': 0  # Adjust as needed
}

def extract_features(form_data):
    feature_values = []
    for feature in features:
        value = form_data.get(feature, default_values.get(feature, 0))
        try:
            feature_values.append(float(value))
        except ValueError:
            flash(f"Invalid input for {feature}. Veuillez entrer une valeur numérique.")
            return None
    return feature_values

@app.route('/')
def home():
    return render_template('select_model.html')

@app.route('/select_model', methods=['POST'])
def select_model():
    selected_model = request.form.get('model')
    if selected_model not in models:
        flash("Le modèle sélectionné n'est pas disponible.")
        return redirect(url_for('home'))
    session['selected_model'] = selected_model
    return redirect(url_for('predict'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    selected_model = session.get('selected_model')
    if not selected_model:
        flash("Veuillez d'abord sélectionner un modèle.")
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        model = models.get(selected_model)
        
        if not model:
            flash("Le modèle sélectionné n'est pas disponible.")
            return redirect(url_for('predict'))
        
        # Extract features from form and make prediction
        feature_values = extract_features(request.form)
        if feature_values is None:
            return redirect(url_for('predict'))  # Extraction failed
        
        try:
            prediction = model.predict([feature_values])
            prediction_result = prediction[0]
            return render_template('result.html', prediction=prediction_result)
        except Exception as e:
            flash(f"Erreur lors de la prédiction: {e}")
            return redirect(url_for('predict'))
    
    return render_template('index.html', features=features, default_values=default_values)

if __name__ == '__main__':
    app.run(debug=True)