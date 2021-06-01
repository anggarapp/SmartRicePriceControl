import h5py
import gcsfs
import numpy
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from flask import Flask, request, jsonify

model_location = 'gs://rice_price_dev/ml_models/test_model_r5.h5'
input_fit_data = [[0,0,0,0],[9,9,9,9]] ## need to change this
output_fit_data = [[1637.76],[115862.16]]
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()
input_scaler.fit(input_fit_data)
output_scaler.fit(output_fit_data)

app = Flask(__name__)

@app.route('/api/price_predict', methods =['GET'] )
def predict():
    try:
        data1 = float(request.args.get('param1'))
        data2 = float(request.args.get('param2'))
        data3 = float(request.args.get('param3'))
        data4 = float(request.args.get('param4'))  
        data = [[data1,data2,data3,data4]]
    except:
        data =[[]]
    results = {
        "Produksi":0.0,
    }
    print(data)

    ##checking input
    if len(data[0]) != 4:
        return jsonify(results)
    for nilai in data[0]:
        if str(type(nilai)) != "<class 'float'>":
            return jsonify(results)

    FS = gcsfs.GCSFileSystem(project='Smart Food Prices Control', token='test_assets/cred.json')
    
    with FS.open(model_location, 'rb') as model_file:
        model_gcs = h5py.File(model_file, 'r')
        test_Model = keras.models.load_model(model_gcs)
    ##
    input_data = input_scaler.transform(data)
    raw_prediction = test_Model.predict(input_data)
    prediction = output_scaler.inverse_transform(raw_prediction)

    results["Produksi"] = float(prediction[0][0])
    
    json_results = jsonify(results)
    return json_results

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='8080',debug=True)