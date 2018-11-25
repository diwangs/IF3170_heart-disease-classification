from flask import Flask, request, render_template
from joblib import load
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_DATA_PATH = '../data/raw'

app = Flask(__name__)

def preprocess_input(input_value):
    train_df = pd.read_csv('{}/tubes2_HeartDisease_train.csv'.format(RAW_DATA_PATH)).drop('Column14', axis=1)
    train_df.loc[train_df.index.max() + 1]  = input_value
    
    cat_idx = [1, 2, 5, 6, 8, 10, 12]
    for i in cat_idx:
        train_df.iloc[-1, i] = np.int64(train_df.iloc[-1, i])
    
    train_columns_replacement = {
        'Column1': 'age',
        'Column2': 'sex',
        'Column3': 'chest_pain_type',
        'Column4': 'resting_blood_pressure',
        'Column5': 'serum_cholestrol',
        'Column6': 'fasting_blood_sugar',
        'Column7': 'resting_ECG',
        'Column8': 'max_heart_rate_achieved',
        'Column9': 'excercise_induced_angina',
        'Column10': 'ST_depression',
        'Column11': 'peak_exercise_ST_segment',
        'Column12': 'num_of_major_vessels',
        'Column13': 'thal',
    }
    train_df = train_df.rename(columns=train_columns_replacement).replace('?', np.NaN)

    def stringify_categorical_data(df, colnames):
        for colname in colnames:
            df[colname] = df[colname].astype(str)

    def impute_numerical_data(df,colnames):
        for colname in colnames:
            df[colname].fillna(df.loc[df[colname]!=np.NaN][colname].median(),inplace=True)
            df[colname] = df[colname].astype(np.float64)

    categorical_data_colname = ['peak_exercise_ST_segment','excercise_induced_angina','thal']
    numerical_data_colname = ['max_heart_rate_achieved',
                              'ST_depression',
                              'num_of_major_vessels',
                              'resting_blood_pressure',
                              'fasting_blood_sugar',
                              'serum_cholestrol',
                              'resting_ECG']


    impute_numerical_data(train_df, numerical_data_colname)
    stringify_categorical_data(train_df, categorical_data_colname)

    train_df = pd.get_dummies(train_df)

    train_features = train_df.values[:-1]

    scaler = StandardScaler().fit(train_features)

    input_features = scaler.transform(train_df.values[-1:])

    return input_features

def predict_result(data):
    list_data = [float(x[0]) for x in data.values()]
    x = preprocess_input(list_data)
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