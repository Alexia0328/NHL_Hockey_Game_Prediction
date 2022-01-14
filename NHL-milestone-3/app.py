import sys
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort, render_template, url_for, redirect
import pandas as pd
import pickle 
from comet_ml import API
from dotenv import load_dotenv 
load_dotenv()
import json


LOG_FILE = os.environ.get('LOG_FILE')

app = Flask(__name__)


model = None
api = API(api_key=os.environ.get('COMET_API_KEY'))


@app.before_first_request
def before_first_request():
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(LOG_FILE)
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p'))
    logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    
    app.logger.info("INFO   | Executed app.before_first_request")
    try: 
        base_model = pickle.load((open("./models/xgb_lg_f1_25.pkl", "rb")))
        app.logger.info("INFO   | Loaded base(default) model: xgboost")

        global model 
        model = base_model 

    except OSError as err:
        app.logger.error(f'ERROR  |  {err} \n occured while loading xgb_lg_f1_25.pkl from local') 


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    try: 
        log_file = open('./flask.log', 'r')
        app.logger.info("INFO   | loaded the logs file")
        response = log_file.read().splitlines()
        log_file.close()
        #return {"data": response}
        return jsonify(response)
    except OSError as err: 
        app.logger.error(f'ERROR  | {err}')
        error_response = "500: Could not load the internal logs file"
        return jsonify({'error': error_response})
    



@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    j = json.loads(request.get_json())
    app.logger.info(f'INFO   | Input json: {j}')

    filename = {'random_forest': 'random_forest_300_6_10.sav', 
              'xgb': 'xgb_lg_f1_25.pkl'}

    model_name = {'random_forest': 'random-forest-300-6-10', 
                  'xgb': 'xgb-optimum'}

    model_arg = j['target_model']
    app.logger.info(f"INFO  | Target model: {model_arg}")
    file = filename[model_arg]
    app.logger.info(f"INFO  | Target file: {file}")

    filepath = './models/'
    global model

    if os.path.exists(filepath + file): 
        app.logger.info(f"INFO  | model: {model_arg} is already downloaded")
        model = pickle.load(open(filepath + file, 'rb'))
        app.logger.info(f"INFO  | successfully loaded model {model_arg} from local")
        response = f'model: {model} is ready for prediction' 
        return jsonify(response)

    else: 
        app.logger.info('INFO  | model is not locally available, downloading it from comet...')
        try: 
            api.download_registry_model(
                                    workspace= j['workspace'], 
                                    registry_name= model_name[model_arg],
                                    version= "1.0.0",
                                    output_path= filepath, 
                                    expand=True)
            app.logger.info(f"INFO  | successfully downloaded {model_arg} from comet")

                
            try: 
                model = pickle.load(open(filepath + file, 'rb'))
                app.logger.info(f'INFO   | successfully loaded model {model_arg} from local')
                response = f'model: {model} is ready for prediction'
                app.logger.info(f'INFO   | {response}')
                return jsonify(response)

            except Exception as err: 
                app.logger.error(f"ERROR  | {err} while loading mdodel: {model_arg} from local")
                error_response = f'500: Error while loadinng {model_arg} from local'
                return jsonify({'error': error_response})

        except Exception as err: 
            app.logger.error(f"ERROR   | {err} while downlaoding {model_arg} from comet")
            error_response = f'500: Could not download {model} from comet'
            return jsonify({'error': error_response})



@app.route("/predict", methods=["POST"])
def predict():
    j = request.get_json()
    #app.logger.info(f'INFO   | type of payload: {type(j)}')
    #app.logger.info(f'INFO   | received dataframe: {j}')

    test = pd.read_json(j)
    #app.logger.info("INFO   | converted json to dataframe")

    try: 
        global model 
        pred = model.predict(test)
        probs = model.predict_proba(test)
        #app.logger.info("INFO  | successfully obtained prediction and probabilities")
        
        response = dict()
        for i, pred in enumerate(pred): 
            response[i] = {'prediction': int(pred), 
                           'goal confidence': float(round(probs[i][1], 3))}


        #app.logger.info(f'INFO   | Prediction response: {response}')
        return jsonify(response)  # response must be json serializable!

    except Exception as err: 
        error_response = f'500 | Could not make a prediction'
        app.logger.error(f'ERROR  | {err}')
        return jsonify({'error': error_response})

if __name__=="__main__":
    app.run(debug=True)