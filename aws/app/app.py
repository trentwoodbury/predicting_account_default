from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = 'something_secret'

# Load the pickled model.
MODEL = pickle.load(open('model.pkl', 'rb'))

@app.route('/api', methods=['GET'])
def api():
    """Handle request and output model score in json format."""
    # Handle empty requests.
    if not request.json:
        return jsonify({'error': 'no request received'})

    # Parse request args into feature array for prediction.
    x_list = parse_args(request.json)
    x_array = np.array([x_list])

    # Predict on x_array and return JSON response.
    estimate = float(MODEL.predict_proba(x_array)[:, 1])
    response = dict(ESTIMATE=estimate)

    return jsonify(response)

def parse_args(request_dict):
    """Parse model features from incoming requests formatted in JSON."""

    # Parse out the features from the request_dict.
    x_list = request_dict['payload']
    return x_list


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
