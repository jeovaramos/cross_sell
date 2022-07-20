import pickle
import pandas as pd
from flask import Flask, request
from flask.wrappers import Response
from data_pipeline.health_insurance import HealthInsurance

# loading model
model = pickle.load(
    open('models/rf_model.pkl', 'rb'))

# initialize API
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def health_insurance_predict():
    test_json = request.get_json()
    print("got json")

    if test_json:  # there is data
        if isinstance(test_json, dict):  # unique example
            test_raw = pd.DataFrame(test_json, index=[0])
            print("got data")

        else:  # multiple example
            test_raw = pd.DataFrame(
                test_json, columns=test_json[0].keys())
            print("got multiple data")

        # Instantiate Rossmann class
        pipeline = HealthInsurance()
        print("instantiated pipeline")

        # data cleaning
        print("data cleaning")
        df1 = pipeline.data_cleaning(test_raw)

        # feature engineering
        print("feature engineering")
        df2 = pipeline.feature_engineering(df1)

        # data preparation
        print("data preparation")
        df3 = pipeline.data_preparation(df2)

        # prediction
        print("prediction")
        df_response = pipeline.get_prediction(model, test_raw, df3)

        return df_response

    else:
        return Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    # port = int(os.environ.get('PORT', 5000))
    app.run('0.0.0.0', debug=True, port=5001)
