from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('serum.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['usia']
    data2 = request.form['mk1']
    data3 = request.form['mk2']
    data4 = request.form['tipe']
    data5 = request.form['sensitif']
    arr = np.array([[data1, data2, data3, data4, data5]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)