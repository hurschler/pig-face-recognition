from flask import Flask, jsonify, request, render_template, redirect, abort, send_from_directory, Response
from PIL import Image as Pil_Image
from apispec import APISpec
from flask_bootstrap import Bootstrap
from flask_apispec.extension import FlaskApiSpec
import marshmallow as ma
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



@app.route("/api/swagger.json")
def create_swagger_spec():
    return jsonify(APISpec.spec.to_dict())


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            img = cv2.imdecode(np.fromstring(request.files["image"].read(), np.uint8), cv2.IMREAD_UNCHANGED)

            image = request.files["image"]
            print(image)
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            return redirect(request.url)
    return render_template("upload_image.html")


@app.route("/files")
def list_files():
    """Endpoint to list files on the server."""
    files = []
    for filename in os.listdir(IMAGE_UPLOADS):
        path = os.path.join(IMAGE_UPLOADS, filename)
        if os.path.isfile(path):
            files.append(filename)
    return jsonify(files)


@app.route("/files/<path:path>")
def get_file(path):
    """Download a file."""
    return send_from_directory(IMAGE_UPLOADS, path, as_attachment=True)


@app.route("/index.html")
def main():
    return render_template('index.html')


@app.route("/files/<filename>", methods=["POST"])
def post_file(filename):
    """Upload a file."""

    if "/" in filename:
        # Return 400 BAD REQUEST
        abort(400, "no subdirectories allowed")

    with open(os.path.join("./upload", filename), "wb") as fp:
        fp.write(request.data)

    # Return 201 CREATED
    return "", 201

@app.route('/api/postimage', methods=['POST'])
def post_image():
    # read image file string data
    filestr = request.files['file'].read()
    # convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('messigray3.png',img)

    # do some fancy processing here....
    # pil_img = Pil_Image.fromarray(im_rgb)
    # pil_img.save(os.path.join(app.config["IMAGE_UPLOADS"], "myUpload3.jpg"))
    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# route http posts to this method
# @api.route('/api/test')
@app.route('/api/postimage', methods=['POST'])
def post_image_v2():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # do some fancy processing here....
    pil_img = Pil_Image.fromarray(im_rgb)
    pil_img.save(os.path.join(app.config["IMAGE_UPLOADS"], "11.jpg"))
    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# @use_kwargs(ImageSchema)
# @marshal_with(ImageSchema, code=201)
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

    print("postjsonimage Foto :", datetime.now())
    response = {'message': 'image received', 'imageId': imageId}
    response_pickled = jsonpickle.encode(response)
    response = Response(response=response_pickled, status=200, mimetype="application/json")
    try:
        os.remove( '../output/' + img_file_name)
    except:
        print("error on file read file: " + img_file_name)

    return response


@app.route('/api/getimage')
def get_image():
    return send_file('../sample/DSC_V1_6460_2238.mask.png', attachment_filename='DSC_V1_6460_2238.mask.png', mimetype='image/png')


@app.route('/api/getimagejson')
def get_image_json():
    imageId = request.args.get('imageId')
    imageType = request.args.get('imageType')
    print("start segemntation ", imageId)
    image_path = '../output/' + imageId +'.' + imageType  # point to your image location
    while True:
        if os.path.isfile(image_path):
            print ("File exist:", image_path)
            try:
                encoded_img = get_response_image(image_path)
                break
            except IOError:
                print ("IOError: " + image_path)
        else:
            print ("File not exist")
            time.sleep(0.5) # Delay for 1 minute (60 seconds).

    numberOfPigs = 1
    imageId = 42
    response_json =  { 'status' : 'Success', 'numberOfPigs': numberOfPigs , 'imageId': imageId, 'imageBytes': encoded_img}
    response_pickled = jsonpickle.encode(response_json)
    response = Response(response=response_pickled, status=200, mimetype="application/json")
    os.remove(image_path)
    # response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route('/api/recognizepig')
def recognize_pig():
    
    return ''



# @use_kwargs(ImageSchema)
# @marshal_with(ImageSchema)
@app.route('/api/sendpet', methods=['POST'])
def put(**kwargs):
    print('pet erhalten')
    print(**kwargs)
    return Response(response='pet', status=200, mimetype="application/json")

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
    docs.register(put)
    app.run(host= '0.0.0.0', port=8080)

