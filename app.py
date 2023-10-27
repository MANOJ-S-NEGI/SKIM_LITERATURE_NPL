from flask import Flask, render_template, request, json
import json
from components.predictions import *

app = Flask(__name__)

print('initiating')


@app.route('/', methods=['GET'])
def Home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    input_data = [{'abstract': request.form['textarea']}]
    prediction_val, abstract_lines = training_prediction(input_data)

    # Turn prediction class integers into string class names
    text_class = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
    response = []
    for i in range(len(prediction_val)):
        line = abstract_lines[i]
        prediction = text_class[prediction_val[i]]
        response.append(f"{prediction} : {line}")
    response_text = "<br>".join(response)

    return render_template('final.html', prediction_texts=response_text)
    # return response_text


if __name__ == '__main__':
    app.run()
