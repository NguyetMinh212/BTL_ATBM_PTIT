
from flask import Flask
from flask import request
from flask import jsonify
from predict import Predictor
import uuid

app = Flask(__name__)
predictor = Predictor()

#Config port for the server
app.config['PORT'] = 8211


@app.route('/predict', methods=['POST'])
def predict():
    # Generate a unique request ID
    request_id = str(uuid.uuid4())
    

    # Get the URL from the request data
    data = request.get_json()
    url = data.get('url', '')

    # Predict whether the URL is good or bad
    prediction = predictor.predict_url(url)
    result = 'good' if prediction == 0 else 'bad'

    # Create the response
    response = {
        'meta': {
            'request_id': request_id,
            'status': 'success',
            'message': 'The prediction was successful.'
        },
        'data': {
            'url': url,
            'prediction': result
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=app.config['PORT'])