from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords')

def preprocess(text):
    return ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])

app = Flask(__name__)
model = pickle.load(open('best_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    raw_features = [x for x in request.form.values()]
    preprocessed_features = [preprocess(x) for x in raw_features]
    prediction = model.predict(preprocessed_features)
    probablity = model.predict_proba(preprocessed_features)

    output = prediction[0]
    prob = round(np.max(probablity[0])*100,2)
    
    return render_template('index.html', prediction_text=output, prediction_probablity=prob)


if __name__ == "__main__":
    app.run(debug=True)