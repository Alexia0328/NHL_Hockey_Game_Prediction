import json
import requests
import pandas as pd
import logging


logger = logging.getLogger(__name__)

print("serving_client.py is being excecuted")

class ServingClient:
    def __init__(self, ip: str = "127.0.0.1", port: int = 3000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """


        resp = requests.post(
            "{}/predict".format(self.base_url), 
            json=X.to_json()
        )
        return resp.json()




    def logs(self) -> dict:
        """Get server logs"""

        resp=requests.get(
            "{}/logs".format(self.base_url)
        )

        return resp.json()

    def download_registry_model(self, workspace: str = "maskedviper", model: str = "xgb", version: str = "1.0.0") -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        dict = {
            "workspace": workspace,
            "target_model": model,
            "version": version
        }
        request_json = json.dumps(dict)
        resp = requests.post(
            "{}/download_registry_model".format(self.base_url), 
            json=request_json
        )
        return resp


# test code

# testServingClient=ServingClient()
# # print(testServingClient.logs()) 
# # log is good

# print(testServingClient.download_registry_model())