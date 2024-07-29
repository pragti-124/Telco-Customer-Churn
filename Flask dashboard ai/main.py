import pandas as pd
import pickle
from flask import Flask, render_template, request
from imblearn.over_sampling import SMOTE

app = Flask(__name__)
model_and_scaler = pickle.load(open("random_forest_model.pkl", 'rb'))
model = model_and_scaler["model"]
scaler = model_and_scaler["scaler"]

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/submit', methods=['POST'])
def submit():
    try:
        OnlineSecurity = request.form.get('OnlineSecurity', '')
        Contract = request.form.get('Contract', '')
        TotalCharges = float(request.form.get('TotalCharges', 0))
        tenure = int(request.form.get('tenure', 0))
        MonthlyCharges = float(request.form.get('MonthlyCharges', 0))
        
        # Ensure sqft is parsed as float
        input_data = pd.DataFrame([[OnlineSecurity, Contract, TotalCharges, tenure, MonthlyCharges]],
                                columns=['OnlineSecurity', 'Contract', 'TotalCharges', 'tenure', 'MonthlyCharges'])

        # Handle categorical variables
        input_data['OnlineSecurity'] = input_data['OnlineSecurity'].map({'Yes': 2, 'No': 0,'No internet service':1})
        input_data['Contract'] = input_data['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

        # Make prediction
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Map prediction to a user-friendly message
        if prediction == 1:
                result = "The customer is likely to churn."
        else:
                result = f"The customer is not likely to churn."
        return result
    except Exception as e:
        return str(e)



if __name__ == "__main__":
    app.run(debug=True, port=5001)
