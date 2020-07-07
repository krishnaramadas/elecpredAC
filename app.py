# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 20:03:20 2020

@author: Krishna
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
  

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output ==1:
        return render_template('index.html', prediction_text='Winner')
    else:
        return render_template('index.html', prediction_text='Not Winner')

if __name__ == "__main__":
    app.run(debug=True)


