import requests
import cv2
from jsonpickle import json
import base64

API_URL = 'http://localhost:8080'
test_url = API_URL + '/api/postimage'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}
img = cv2.imread('11.jpg')
_, img_encoded = cv2.imencode('.jpg', img)
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
print(json.loads(response.text))

test2_url = API_URL + '/api/postjsonimage'
with open('image0.jpg', 'rb') as f:
    im_b64 = base64.b64encode(f.read())

payload = {'id': '123.jpg', 'type': 'jpg', 'box': [0, 0, 100, 100], 'image': im_b64}
response = requests.post(test2_url, data=payload)
print(json.loads(response.text))

test3_url = API_URL + '/api/sendpet'
response = requests.post(test3_url, data=payload)
print(json.loads(response.text))