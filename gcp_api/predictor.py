import os
import pickle

import numpy as np
from xgboost import XGBClassifier


class MyPredictor(object):
    """Predictor for Google AI Platform"""


    def __init__(self, model, preprocessor):
        """
        Stores artifacts for prediction. Only initialized via `from_path`.
        """
        self._model = model
        self._preprocessor = preprocessor


    def predict(self, instances, **kwargs):
        """
        Performs custom prediction. Predicts the probability of a default for
        the given instances.

        Args:
            instances: A list of prediction input instances.
            **kwargs: A dictionary of keyword args provided as additional
                fields on the predict request body.

        Returns:
            A list of outputs containing the prediction results.
        """
        inputs = np.asarray(instances)
        outputs = self._model.predict_proba(inputs)[:, 1]
        return outputs.tolist()


    @classmethod
    def from_path(cls, model_dir):
        """Creates an instance of MyPredictor using the given path.

        This loads artifacts that have been copied from your model directory in
        Cloud Storage. MyPredictor uses them during prediction.

        Args:
            model_dir: The local directory that contains the trained
                xgboost model.

        Returns:
            An instance of `MyPredictor`.
        """
        model_path = os.path.join(model_dir, 'model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        preprocessor = None

        return cls(model, preprocessor)
