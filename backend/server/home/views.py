from django.http import JsonResponse
from django.shortcuts import render
from .ml_models import random_forest_model, extra_trees_model
import pandas as pd

# Create your views here.
def predict(request):
    response=None
    if request.method == 'POST':
        # Get the data from the POST request
        data = request.POST
        
        # Extract features from the data
        features = [
            float(data['Pregnancies']),
            float(data['Glucose']),
            float(data['BloodPressure']),
            float(data['SkinThickness']),
            float(data['Insulin']),
            float(data['BMI']),
            float(data['DiabetesPedigreeFunction']),  # This feature was missing
            float(data['Age'])
        ]
        print(features)



        # Convert to DataFrame to preserve feature names
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        features_df = pd.DataFrame([features], columns=feature_names)
        print(features_df)

        # Make predictions using both models
        rfc_prediction = int(random_forest_model.predict(features_df)[0])
        rfcet_prediction = int(extra_trees_model.predict(features_df)[0])
        print(rfc_prediction)
        print(rfcet_prediction)

        # Return the results as JSON
        prediction_mapping = {1: 'diabetes', 0: 'non-diabetes'}
        
        rfc_label = prediction_mapping[rfc_prediction]
        rfcet_label = prediction_mapping[rfcet_prediction]

        # Return the results as JSON
        response = {
            'random_forest_prediction': rfc_label,
            'extra_trees_prediction': rfcet_label
        }

    return render(request, 'predict.html', {'response': response})
