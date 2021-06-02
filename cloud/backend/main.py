# input_suhu_ratarata bulan 1
# input_suhu_ratarata bulan 2
# input_suhu_ratarata bulan 3
# input_suhu_ratarata bulan 4
# input_kelembaban_ratarata bulan 1
# input_kelembaban_ratarata bulan 2
# input_kelembaban_ratarata bulan 3
# input_kelembaban_ratarata bulan 4
# input_curah_hujan bulan 1
# input_curah_hujan bulan 2
# input_curah_hujan bulan 3
# input_curah_hujan bulan 4
# input_lama_penyinaran
# input_luas_lahan

import h5py
import gcsfs
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from flask import Flask, request, jsonify


model_location = 'gs://rice_price_dev/ml_models/test_model_r2.h5'
input_fit_data = [[26.49, 26.49, 26.49, 26.49, 67, 67, 67, 67, 0, 0, 0, 0, 3.27, 333],
                  [30.25, 30.25, 30.25, 30.25, 85.45, 85.45, 85.45, 85.45, 18.84, 18.84, 18.84, 21.84, 9.47, 29694]]  # need to change this
output_fit_data = [[1194.46], [103583.39]]
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()
input_scaler.fit(input_fit_data)
output_scaler.fit(output_fit_data)


app = Flask(__name__)


@app.route('/api/price_predict', methods=['GET'])
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
        data10 = float(request.args.get('param10'))
        data11 = float(request.args.get('param11'))
        data12 = float(request.args.get('param12'))
        data13 = float(request.args.get('param13'))
        data14 = float(request.args.get('param14'))
        data = [[data1, data2, data3, data4, data5, data6, data7,
                 data8, data9, data10, data11, data12, data13, data14]]
    except:
        data = [[]]
    results = {
        "Produksi": 0.0,
    }
    print(data)

    # checking input
    if len(data[0]) != 14:
        return jsonify(results)
    for nilai in data[0]:
        if str(type(nilai)) != "<class 'float'>":
            return jsonify(results)

    FS = gcsfs.GCSFileSystem(
        project='Smart Food Prices Control', token='test_assets/cred.json')

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
    app.run(host='0.0.0.0', debug=True, port=int(os.environ.get("PORT", 8080)))
