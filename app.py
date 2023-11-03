# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'Soil-Analysis-Prediction.pkl'
rf = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        pH = float(request.form['pH'])
        EC = float(request.form['EC'])
        OC = float(request.form['OC'])
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        S = float(request.form['S'])
        Ca = float(request.form['Ca'])
        Mg = float(request.form['Mg'])
        Zn = float(request.form['Zn'])
        Cu = float(request.form['Cu'])
        Fe = float(request.form['Fe'])
        Mn = float(request.form['Mn'])
        B = float(request.form['B'])
        Mo = float(request.form['Mo'])
        Tex = int(request.form['Tex'])

        data = np.array([[
    pH, EC, OC, N, P, K, S, Ca, Mg, Zn, Cu, Fe, Mn, B, Mo, Tex
]])

        my_prediction = rf.predict(data)
        
        predicted_val= [ '%.3f' % my_prediction ]
        predicted_val=float(predicted_val[0])
        return render_template('result.html', prediction=predicted_val)

if __name__ == '__main__':
	app.run(debug=True)