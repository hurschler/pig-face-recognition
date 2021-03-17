from flask import Flask, jsonify, request, render_template, redirect, abort, send_from_directory, Response
from PIL import Image as Pil_Image
from apispec import APISpec
from flask_bootstrap import Bootstrap
from flask_apispec.extension import FlaskApiSpec
from flask import Flask, g, session
from flask import current_app as app
from flask import send_file
from base64 import encodebytes
from apispec.ext.marshmallow import MarshmallowPlugin
from datetime import datetime
from flask_cors import CORS, cross_origin


import flask_apispec
import os
import cv2
import numpy as np
import jsonpickle
import base64
import io
import time
import logging.config
import util.logger_init

from util import logger_init
from flask_apispec import marshal_with, use_kwargs

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

SWAGGER_URL = '/api/docs'
API_URL = "/api/swagger.json"

app = Flask(__name__)
cors = CORS(app)

IMAGE_UPLOADS = "./upload"
app.config["IMAGE_UPLOADS"] = IMAGE_UPLOADS

log = logging.getLogger(__name__)


@app.route("/api/swagger.json")
def create_swagger_spec():
    return jsonify(APISpec.spec.to_dict())


@app.route("/index.html")
def main():
    return render_template('index.html')



@app.route('/api/postjsonimage', methods=['POST'])
def post_json_image():
    payload = request.form.to_dict(flat=False)
    im_b64 = payload['image'][0]  # remember that now each key corresponds to list.
    imageId = payload['imageId'][0]
    imageType = payload['imageType'][0]
    im_binary = base64.b64decode(im_b64)
    buf = io.BytesIO(im_binary)
    pil_img = Pil_Image.open(buf)
    img_file_name = imageId + '.' + imageType
    img_upload_path = os.path.join(app.config["IMAGE_UPLOADS"], img_file_name)
    if imageType == 'jpg' or imageType == 'jpeg':
        pil_img.convert('RGB').save(img_upload_path)
    else:
        pil_img.save(img_upload_path)

    log.info("postjsonimage ImageId: " + imageId + "Type: " + imageType + " " + str(datetime.now()))
    response = {'message': 'image received', 'imageId': imageId}
    response_pickled = jsonpickle.encode(response)
    response = Response(response=response_pickled, status=200, mimetype="application/json")
    try:
        os.remove( '../output/' + img_file_name)
    except:
        log.error("error on file read file: " + img_file_name)

    return response



@app.route('/api/getimagejson')
def get_image_json():
    imageId = request.args.get('imageId')
    imageType = request.args.get('imageType')
    log.info("Server GET wait for segemntation ImageId: " + imageId + " ImageType: " + imageType)
    image_path = '../output/' + imageId +'.' + imageType  # point to your image location
    i = 0
    while True:
        if os.path.isfile(image_path):
            log.info("File exist: " + image_path)
            try:
                encoded_img = get_response_image(image_path)
                break
            except IOError:
                log.error("IOError: " + image_path)
        else:
            i = i +1
            log.info("File not exist: " + image_path)
            time.sleep(0.2) # Delay for 1 minute (60 seconds).

    numberOfPigs = 1
    response_json =  { 'status' : 'Success', 'numberOfPigs': numberOfPigs , 'imageId': imageId, 'imageBytes': encoded_img}
    response_pickled = jsonpickle.encode(response_json)
    response = Response(response=response_pickled, status=200, mimetype="application/json")
    os.remove(image_path)
    # response.headers.add("Access-Control-Allow-Origin", "*")
    return response


def get_response_image(image_path):
    pil_img = Pil_Image.open(image_path, mode='r') # reads the PIL image
    basewidth = 300
    wpercent = (basewidth/float(pil_img.size[0]))
    hsize = int((float(pil_img.size[1])*float(wpercent)))
    img = pil_img.resize((basewidth,hsize), Pil_Image.ANTIALIAS)

    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

def get_response_image_from_pil(pil_img):
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img


if __name__ == "__main__":
    Bootstrap(app)
    app.config.update({
        'APISPEC_SPEC': APISpec(
            title='pig_face_detection',
            version='v1',
            openapi_version='3.0.2',
            plugins=[MarshmallowPlugin()],
        ),
        'APISPEC_SWAGGER_URL': '/swagger/',
    })
    docs = FlaskApiSpec(app)
    docs.register(post_json_image)
    app.run(host= '0.0.0.0', port=8080)

