# app.py
from flask import Flask, render_template, request
from backend import predict_iris

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        prediction = predict_iris(sepal_length, sepal_width, petal_length, petal_width)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)