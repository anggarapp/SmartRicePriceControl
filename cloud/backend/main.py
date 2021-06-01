import h5py
import gcsfs
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from flask import Flask, request, jsonify

model_location = 'gs://rice_price_dev/ml_models/test_model_r1.h5'
input_fit_data = [[0,0,0,25.67,65.7,0,2.91,411],[1,1,1,30.25,87.5,22.05,9.73,30824]]
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
        data5 = float(request.args.get('param5'))
        data6 = float(request.args.get('param6'))
        data7 = float(request.args.get('param7'))
        data8 = float(request.args.get('param8'))
        data9 = float(request.args.get('param9'))   
        data = [[data1,data2,data3,data4,data5,data6,data7,data8]]
    except:
        data =[[]]
    results = {
        "Produksi":0.0,
        "Saran":"NULL"
    }
    print(data)

    ##checking input
    if len(data[0]) != 8:
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
    selisih = prediction[0][0] - data9
    if (selisih > 0):
        results["Saran"] = ("anda disarankan mendistribusikan beras sebesar " +str(abs(selisih))+" ton ke wilayah lain")
    elif (selisih < 0):
        results["Saran"] = ("anda disarankan mendapat distribusi beras sebesar " +str(abs(selisih))+" ton dari wilayah lain")
    else:
        results["Saran"] = ("stok beras wilayah anda seimbang")

    json_results = jsonify(results)
    return json_results

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=int(os.environ.get("PORT", 8080)),debug=True)