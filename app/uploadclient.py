import requests
import cv2
import os
from jsonpickle import json
import base64
import util.config as config

API_URL = 'http://localhost:8080'
SAMPLE_IMG = '../sample/DSC_V1_6460_2238.JPG'
new_pig_name = 6950
URL_POST_JSON_IMAGE_DETECTION = API_URL + '/api/postjsonimage'
URL_GET_JSON_IMAGE_DETECTION = API_URL + '/api/getimagejson'
URL_GET_JSON_IMAGE_RECOGNITION = API_URL + '/api/getrecognition'
URL_POST_NEW_PIG_IMAGE = API_URL + '/api/postnewpigimage'
URL_POST_NEW_PIG_FINISHED = API_URL + '/api/postnewpigfinished'

PIG_RECOGNITION = False
ADD_NEW_PIG = True

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

# --- Detection & Recognition of a Pig ---------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
if PIG_RECOGNITION:
    with open(SAMPLE_IMG, 'rb') as f:
        im_b64 = base64.b64encode(f.read())

    # Post Image for detection
    payload_detection_post = {'imageId': '123', 'imageType': 'jpg', 'image': im_b64}
    response_detection = requests.post(URL_POST_JSON_IMAGE_DETECTION, data=payload_detection_post)
    print('response detection post: ' + str(json.loads(response_detection.text)))

    # Get Detection Information & Image with mask
    payload_detection_get = {'imageId': '123', 'imageType': 'jpg'}
    response_detection = requests.get(URL_GET_JSON_IMAGE_DETECTION, data=payload_detection_get)
    print('response detection get: ' + str(json.loads(response_detection.text)))

    # Get Recognition Result Information
    # Example {'imageId': 'DSC_V2_6446_2774.JPG-crop-mask0.jpg', 'pig_name': '6446', 'accuracy': '0.9731434'}
    payload_recognition_get = {'imageId': '123', 'imageType': 'jpg'}
    response_recognition = requests.get(URL_GET_JSON_IMAGE_RECOGNITION, data=payload_recognition_get)
    print('response recognition get: ' + str(json.loads(response_recognition.text)))

# --- Add a new Pig ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
if ADD_NEW_PIG:
    new_pig_path_train = config.image_new_pig_path_train
    new_pig_path_train = os.path.join(new_pig_path_train, str(new_pig_name))
    image_names = os.listdir(new_pig_path_train)
    for image_name in image_names:
        with open(os.path.join(new_pig_path_train,image_name), 'rb') as f:
            im_b64 = base64.b64encode(f.read())
            imageId = os.path.splitext(image_name)[0]
            payload_detection_post = {'imageId': imageId, 'imageType': 'jpg', 'pigName': new_pig_name, 'image': im_b64}
            response_add_new_pig_image = requests.post(URL_POST_NEW_PIG_IMAGE, data=payload_detection_post)
            print('response add new pig image post: ' + str(json.loads(response_add_new_pig_image.text)))

    payload_new_pig_finished_post = {'pigName': new_pig_name}
    response_new_pig_finished = requests.post(URL_POST_NEW_PIG_FINISHED, data=payload_new_pig_finished_post)
    print('response new pig finished post: ' + str(json.loads(response_new_pig_finished.text)))

