from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

clf = joblib.load('cancer.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None

    if request.method == 'POST':
        user_data = pd.DataFrame(columns=['GENDER', 'AGE', 'SMOKING',
                                           'YELLOW_FINGERS', 'ANXIETY',
                                           'PEER_PRESSURE', 'CHRONIC_DISEASE',
                                           'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING','SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'])

        for column in user_data.columns:
            user_input = request.form.get(column)
            user_data.at[0, column] = float(user_input)

        user_data = user_data.apply(pd.to_numeric, errors='coerce')

        # Normalize the user data using the fitted scaler
        user_data_scaled_array = scaler.transform(user_data.values.reshape(1, -1))

        user_prediction = clf.predict(user_data_scaled_array)
        prediction = 'Cancer' if user_prediction[0] == 1 else 'Not Cancer'

    return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

