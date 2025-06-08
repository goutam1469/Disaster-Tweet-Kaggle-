from flask import Flask, jsonify, request, render_template
import pickle
import nltk
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
xgb_best = pickle.load(open('xgb_best.pkl','rb'))

nltk.download('punkt')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():    
    text = request.form.get('Tweet')
    tokens = word_tokenize(text.lower())
    vector = model.infer_vector(tokens).reshape(1,-1)

    #predict
    prediction = xgb_best.predict(vector)[0]

    return render_template('index.html', prediction = prediction)
    # return jsonify({'text': text, 'prediction' : 'prediction' if prediction == 1 else 'not_propaganda'})

if __name__ == '__main__':
    app.run(debug=True)

