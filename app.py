from flask import Flask
from flask import request, url_for, redirect, render_template, jsonify

from pycaret.regression import *
import pandas  as pd
import pickle
import numpy as np


app = Flask(__name__)

model = load_model('SocialAnalytics-master')
cols =['Actual_Pos_Score','Actual_neu_Score','Actual_neg_Score','Compound_Score']

@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/Distinguishing Sentiments', methods = ['POST'])  

def predict():

    init_features = [x for x in request.form.values()]
    final = np.array (init_features)
    data_unseen = pd.DataFrame([Final], columns = cols)
    prediction = predict_model(model, data= data_unseen, round=0)
    prediction = int(prediction.label[0])

    return render_template('home.html', pred = 'Expect will be {}'.format(prediction)) 

