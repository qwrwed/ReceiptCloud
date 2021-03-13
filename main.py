# # #   https://www.smashingmagazine.com/2020/08/api-flask-google-cloudsql-app-engine/
# https://cloud.google.com/appengine/docs/standard/python3/building-app/writing-web-service

# main.py
from flask import Flask, request, Response, send_file, make_response, jsonify, render_template
import numpy as np
#from matplotlib import pyplot as plt
#import matplotlib
import cv2
#import jsonpickle
import json
import datetime
import base64
import sys
import importlib
import socket
import ip
#import pytesseract
import utils
#from pytesseract import Output
import cloud

app = Flask(__name__)

@app.route('/')
def root():
    #return 'Please use the ReceiptCapture app at https://mega.nz/file/BQI3SQQZ#3zXmwic4ZnAGK5kIMugyp3Qxdqmhu5FInAdILnTOtWI to access this service.'
    return render_template('index.html')


@app.route('/hello/<name>')
def hello_name(name=""):
    return f'Hello {name}!'

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify(message="Internal Server Error"), 500

@app.route('/upload', methods=["POST"])
def get_upload():
    print("GET UPLOAD")
    args = json.loads(request.form['args'])
    will_return_img = args['returnImg']
    """
    info = "a\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\na\n"
    response = {
        'info': info,
    }
    return response, 200
    """
    importlib.reload(sys.modules.get('pipeline', sys))
    #from preprocessing import process

    img = cv2.imdecode(np.frombuffer(request.files['file'].read(), np.uint8), cv2.IMREAD_COLOR)
    now = datetime.datetime.now()
    now_string = now.strftime('%Y%m%d%H%M%S')
    #cv2.imwrite(f'Datasets/received/received_{now_string}.jpg', img)
    
    path_img_tmp = f'received.jpg'
    cv2.imwrite(path_img_tmp, img)
    data_raw = cloud.image_to_data(path_img_tmp, cloud.FeatureType.BLOCK)
    info = cloud.data_raw_to_fulltext(data_raw)
    #info = "Sample Info"
    
    img = ip.color(img)
    z = img.copy()
    z[:] = 0
    img = cv2.addWeighted(img, 0.5, z, 0.5, 0)        
    
    #img = ip.draw_data(img, data, True, True)
    
    img_out = img
    #info, img_out = process(img, will_return_img)
    response = {
        #'info': info,
        'info': info,
    }
    
    if will_return_img:
        cv2.imwrite("SAVED.jpg", will_return_img)
        img_out_b64 = base64.b64encode(cv2.imencode('.jpg', img_out)[1]).decode()
        response['image'] = img_out_b64

    return response, 200
    

# if __name__ == '__main__':


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    #app.run(host='127.0.0.1', port=8080, debug=True)
    utils.show_address()
    app.run("0.0.0.0", debug=True)