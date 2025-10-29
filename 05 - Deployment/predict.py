import pickle

from flask import Flask
from flask import request, jsonify

C=1.0
model_file = f'model_C={C}.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# customer = {
#     'gender': 'female',
#     'seniorcitizen': 0,
#     'partner': 'yes',
#     'dependents': 'no',
#     'phoneservice': 'no',
#     'multiplelines': 'no_phone_service',
#     'internetservice': 'dsl',
#     'onlinesecurity': 'no',
#     'onlinebackup': 'yes',
#     'deviceprotection': 'no',
#     'techsupport': 'no',
#     'streamingtv': 'no',
#     'streamingmovies': 'no',
#     'contract': 'month-to-month',
#     'paperlessbilling': 'yes',
#     'paymentmethod': 'electronic_check',
#     'tenure': 1,
#     'monthlycharges': 29.85,
#     'totalcharges': 29.85
# }

app = Flask('churn')
@app.route('/predict', methods=['POST']) # POST porque enviamos información en el body
def predict():

    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5
    result = {'churn_probability': float(y_pred),
                'churn': bool(churn)
    }
    return jsonify(result)

# print('input', customer)
# print('churn probability', y_pred)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)