from flask import Flask, request, render_template
from joblib import load
import numpy as np

app = Flask(__name__)

# load best model

def predict_result(data):
    list_data = list(map(float, data.values()))
    x = np.reshape(list_data, (1, -1))
    y = best_model.predict(x)

    result = {
        "verdict": ("Normal" if y == 0 else "Disease"),
        "category": y
    }

    return result

@app.route('/', methods = ['GET', 'POST'])
def render_home():
    if (request.method == "GET"):
        return render_template('home.html', result = None)
    else:
        data = dict(request.form)
        result = predict_result(data)

        return render_template('home.html', result = result)

if __name__ == '__main__':
    global best_model
    best_model = load('../models/best_model.pkl')

    app.run(debug = True, port=3000, host='0.0.0.0')