import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow.keras.backend as K
import logging.config
import util.logger_init
import util.detection_config as detection_config
import jsonpickle
import recognition.ml_data


def convert_to_json_and_save(ml_data):
    ml_data_json = jsonpickle.encode(ml_data)
    with open('../output/data.json', "w") as fh:
        fh.write(ml_data_json)


def load_ml_data_from_json_file(ml_data, file_full_name):
    with open(file_full_name, "r") as fh:
        ml_data = jsonpickle.loads(fh.read())
    return ml_data


def calculate_feature_vectors_train(facenet_face_model, ml_data):
    facenet_face_model.get_embeddings_train(ml_data)


def calculate_feature_vectors_test(facenet_face_model, ml_data):
    facenet_face_model.get_embeddings_test(ml_data)

def plot(img):
    plt.figure(figsize=(8, 4))
    plt.imshow(img[:, :, ::-1])
    plt.axis('off')
    plt.show()


def predict2(facenet_face_model, classification_model, ml_data, img_name):
    width = 160
    height = 160
    img_pil = load_img(img_name, target_size=(width, height))

    if img_pil is None or img_pil.size == 0:
        print("Please check image path or some error occured")
    else:
        persons_in_img = []
        img_encode = facenet_face_model.get_embedding(img_name)
        # Make Predictions
        print ('pig_name: ', img_name, 'length of Feature-Vector: ', len(img_encode), ' Feature-Vector: ', img_encode)
        name = classification_model.predict2(img_encode, 0,0,width, height, ml_data.pig_dict, img_pil)
        persons_in_img.append(name)
        # Save images with bounding box,name and accuracy
        img_opencv = np.array(img_pil)
        img_opencv = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
        cv2.imwrite('../output/recognized_img.jpg', np.array(img_opencv))
        # Pig in image
        print('Pig(s) in image is/are:' + ' '.join([str(elem) for elem in persons_in_img]))

        return img_opencv
