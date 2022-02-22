from flask import Flask, request
from flask_cors import CORS

from user_reviews import *

app = Flask(__name__)
CORS_ALLOW_ORIGIN = "*,*"
CORS_EXPOSE_HEADERS = "*,*"
CORS_ALLOW_HEADERS = "content-type,*"
cors = CORS(app, origins=CORS_ALLOW_ORIGIN.split(","), allow_headers=CORS_ALLOW_HEADERS.split(","),
            expose_headers=CORS_EXPOSE_HEADERS.split(","), supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'
model = keras.models.load_model('my_model')


@app.route('/')
def hello_world():
    print('hello')
    return 'Hello World!'


@app.route('/getReview', methods=['POST'])
def get_review():
    json_request = request.get_json()
    review = json_request['review']
    return str(rate_single_review(review, model))


if __name__ == '__main__':
    app.run(host='0.0.0.0')
