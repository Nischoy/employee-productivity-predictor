from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('XGBoost.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        try:
            features = [
                float(request.form['quarter']),
                float(request.form['department']),
                float(request.form['day']),
                float(request.form['team']),
                float(request.form['targeted_productivity']),
                float(request.form['smv']),
                float(request.form['over_time']),
                float(request.form['incentive']),
                float(request.form['idle_time']),
                float(request.form['idle_men']),
                float(request.form['no_of_workers']),
                float(request.form['no_of_style_change']),
                float(request.form['month']),
            ]

            final_features = [np.array(features)]
            prediction = model.predict(final_features)

            return render_template('submit.html', prediction_text=f'Predicted Productivity: {prediction[0]:.4f}')

        except Exception as e:
            return render_template('submit.html', prediction_text=f'Error in input: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
