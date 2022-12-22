from flask import Flask, request, jsonify, session, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import r2_score
from flask_cors import CORS
import pandas.api.types
import os
import shutil
import joblib
import warnings
np.random.seed(1)

warnings.warn('ignore', category=FutureWarning)

app = Flask(__name__)
cors = CORS(app)
ALLOWED_EXTENSIONS = (['csv'])
app.secret_key = "abcdef"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/load_data', methods=["GET", "POST"])
def data_load():
    if request.method == 'POST':
        UPLOAD_FOLDER_1 = 'data/'
        UPLOAD_FOLDER_2 = 'models/'
        if os.path.isdir(UPLOAD_FOLDER_1):
            shutil.rmtree('data')
        if os.path.isdir(UPLOAD_FOLDER_2):
            shutil.rmtree('models')
        if not os.path.isdir(UPLOAD_FOLDER_1):
            os.mkdir(UPLOAD_FOLDER_1)
        if not os.path.isdir(UPLOAD_FOLDER_2):
            os.mkdir(UPLOAD_FOLDER_2)

        app.config['UPLOAD_FOLDER_1'] = UPLOAD_FOLDER_1
        if 'file' not in request.files:
            return jsonify('No file part')
        file = request.files['file']
        if file.filename == '':
            return jsonify('No selected file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER_1'], file.filename)
            file.save(filepath)
        else:
            return jsonify('Unsupported File Format')
        df = pd.read_csv(filepath)
        session['filepath'] = filepath
        cols = list(df.columns)
        session['columns'] = cols
        cols_response = {'columns': session['columns']}
        return cols_response


@app.route('/train', methods=["GET", "POST"])
def model_training():
    if request.method == 'POST':
        file = os.listdir('data/')
        dataset = pd.read_csv('data/' + file[0])
        target = request.form['target']

        if pandas.api.types.is_numeric_dtype(dataset[target]):
            dataset['date'] = pd.to_datetime(dataset['date'])
            dataset['month'] = [i.month for i in dataset['date']]
            dataset['year'] = [i.year for i in dataset['date']]
            dataset['day_of_week'] = [i.dayofweek for i in dataset['date']]
            dataset['day_of_year'] = [i.dayofyear for i in dataset['date']]
            dataset['day'] = [i.day for i in dataset['date']]
            X = dataset.drop([target, 'date'], axis=1)
            y = dataset[target]
            train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = ExtraTreesRegressor(n_estimators=100, max_features='auto', verbose=1, n_jobs=1, random_state=1)
            clf.fit(train_x, train_y)
            predicted = clf.predict(test_x)
            error = mean_absolute_error(test_y, predicted)
            r2 = r2_score(test_y, predicted)
            n = len(test_x)
            p = len(test_x.columns)
            Adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            modelDetails = {'MAE': error, 'r2 score': r2, 'Adjusted r2 score': Adj_r2}
            joblib.dump(clf, 'models/' + "model.pkl")
        else:
            return jsonify('Target Feature name should be Numeric')
        return modelDetails


@app.route('/save_model', methods=["GET", "POST"])
def savemodel():
    model_name = request.form['model_name']
    if not os.path.isdir('weights'):
        os.mkdir('weights')
    path = os.listdir('weights/')
    if "model.pkl" not in os.listdir('models/'):
        return jsonify("No weights found to save! Train model before saving!")
    if model_name + ".pkl" not in path:
        shutil.move('models/model.pkl', "weights/")
        str_current_datetime = datetime.now().strftime('%d-%m-%y_%H_%M.pkl')
        new_name = model_name + " " + str_current_datetime
        os.rename('weights/model.pkl', 'weights/' + new_name)
    else:
        return jsonify('Model Name already exists!')
    return jsonify('Model Saved Successfully')


@app.route('/model_list', methods=["GET", "POST"])
def model_list():
    path = os.listdir('weights/')
    lis = []
    for i in range(len(path)):
        lis.append(path[i][:-4])
    available_models = {"Models": lis}
    return available_models


@app.route('/test', methods=['GET', 'POST'])
def model_test():
    if request.method == 'POST':
        UPLOAD_FOLDER = 'test_data/'
        file = request.files['test_file']
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree('test_data')
        if not os.path.isdir(UPLOAD_FOLDER):
            os.mkdir(UPLOAD_FOLDER)
        if 'test_file' not in request.files:
            return jsonify('No file part')
        if file.filename == '':
            return jsonify('No selected file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
        else:
            return jsonify('Unsupported File Format')
        model_name = request.form['model_name']
        data_test = pd.read_csv(filepath)
        data = data_test.copy()
        data['date'] = pd.to_datetime(data['date'])
        data['month'] = [i.month for i in data['date']]
        data['year'] = [i.year for i in data['date']]
        data['day_of_week'] = [i.dayofweek for i in data['date']]
        data['day_of_year'] = [i.dayofyear for i in data['date']]
        data['day'] = [i.day for i in data['date']]
        X = data.drop(["date"], axis=1)
        path = os.listdir('weights/')
        lis = []
        for i in range(len(path)):
            lis.append(path[i][:-4])
        available_models = {"Models": lis}
        if model_name in available_models['Models']:
            model = joblib.load('weights/' + model_name + '.pkl')
        else:
            return jsonify("Model not found!")
        data_test['result'] = model.predict(X)
        data_test['result'] = data_test['result'].apply(np.ceil)
        data_test['result'] = data_test['result'].astype(int)
        if os.path.isfile("Output.csv"):
            os.remove("Output.csv")
        data_test.to_csv("Output.csv", index=False)
        # print(data_test.to_dict())
        return data_test.to_dict()
    else:
        return jsonify('GET method is not supported')


@app.route('/download', methods=['GET', 'POST'])
def download():
    if request.method == 'GET':
        if os.path.isfile("Output.csv"):
            return send_from_directory(os.getcwd(), path="Output.csv", as_attachment=True)
        else:
            return jsonify("No Predictions")


@app.route('/plot_chart', methods=['GET', 'POST'])
def plot_chart():
    df = pd.read_csv("Output.csv")
    df['date'] = pd.to_datetime(df['date'])
    tdf1 = df
    tdf_ = tdf1.groupby('item')['result'].sum()
    tdf_.sort_values(ascending=False, inplace=True)
    ind = list(tdf_.index)[:10]
    dic = {}
    for i in ind:
        tdf = tdf1[tdf1['item'] == i]
        tdf.set_index("date", inplace=True)
        tdf = tdf.resample("W")['result'].sum()
        tdf = tdf.reset_index()
        tdf['date'] = tdf['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        dic["item" + str(i)] = {"date": list(tdf['date']), "sales": list(tdf['result'])}
    return dic


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
