import json
import pickle

import pandas as pd
import xgboost as xgb


def validate_data(model, data_predict):

    data_predict_labels = data_predict.columns
    _feature_names = model.feature_names

    _not_missing_ = []
    _missing = []
    _data = []
    data = []

    for _feature in _feature_names:
        if data_predict_labels.__contains__(_feature):
            _not_missing_.append(_feature)
            _data.append(data_predict[_feature][0])
        else:
            _missing.append(_feature)

    data.append(_data)
    df = pd.DataFrame(data=data, columns=_feature_names)

    return df


def XGBoost_predict(_prediction_data):

    filename = 'output/xgboost-model'

    model = pickle.load(open(filename, 'rb'))

    data_predict = pd.DataFrame.from_dict(_prediction_data)

    df = validate_data(model, data_predict)

    dpredict = xgb.DMatrix(data=df)
    result = model.predict(dpredict)
    print(round(result[0]*10))
    return round(result[0] * 10)  # prediction result
