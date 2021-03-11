import os
# import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import vgg_face_model
import classification_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K
import vgg_face_recognition_utils_v1 as rec_util
from PIL import Image as pil_image

import ml_data

def read_pig_images_from_disk(vgg_face_model, ml_data):
    # Prepare Train Data
    path = '.'
    img_path_crop = '/images_crop'
    pig_img_folders = os.listdir(path + img_path_crop)
    for i, pig_name in enumerate(pig_img_folders):
        ml_data.pig_dict[i] = pig_name
        image_names = os.listdir(path + img_path_crop + '/' + pig_name + '/')
        for image_name in image_names:
            img = load_img(path + img_path_crop + '/' + pig_name + '/' + image_name, target_size=(224, 224))
            # img = load_img(path + img_path_crop + '/' + pig_name + '/' + image_name)
            # img.save(path + '/output/' + image_name, "JPEG")
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            img_encode = vgg_face_model.vgg_face(img)
            ml_data.x_train.append(np.squeeze(K.eval(img_encode)).tolist())
            ml_data.y_train.append(i)
            rec_util.test_data(vgg_face_model, ml_data, i, path, pig_name)


def test_data(vgg_face_model, ml_data, i, path, pig_name):
    # Prepare Test Data
    person_folders = os.listdir(path + '/images_crop_test/')
    test_image_names = os.listdir('images_crop_test/' + pig_name + '/')
    for image_name in test_image_names:
        img = load_img(path + '/images_crop_test/' + pig_name + '/' + image_name, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        img_encode = vgg_face_model.vgg_face(img)
        ml_data.x_test.append(np.squeeze(K.eval(img_encode)).tolist())
        ml_data.y_test.append(i)

def plot(img):
    plt.figure(figsize=(8, 4))
    plt.imshow(img[:, :, ::-1])
    plt.axis('off')
    plt.show()

def detect_face(img):
    global rects
    # Detect Faces
    # dnnFaceDetector = dlib.cnn_face_detection_model_v1("model/mmod_human_face_detector.dat")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # rects = dnnFaceDetector(gray, 1)
    # return rects
    return img


def predict(vgg_face_model, classification_model, ml_data, img_name):
    img = cv2.imread(img_name)
    if img is not None and 0 != img.size:
        persons_in_img = []
        rects = rec_util.detect_face(img)
        left, top, right, bottom = 0, 0, 0, 0
        for (i, rect) in enumerate(rects):
            img_encode = vgg_face_model.get_embeddings(img_name)
            # Make Predictions
            embed = K.eval(img_encode)
            name = classification_model.predict(embed, rect, ml_data.pig_dict, img)
            persons_in_img.append(name)
        # Save images with bounding box,name and accuracy
        cv2.imwrite(os.getcwd() + '/output/recognized_img.jpg', img)
        # Pig in image
        print('Pig(s) in image is/are:' + ' '.join([str(elem) for elem in persons_in_img]))

        return img
    else:
        print("Please check image path or some error occured")


def predict2(vgg_face_model, classification_model, ml_data, img_name):
    img = cv2.imread(img_name)

    width = img.shape[1]
    height = img.shape[0]

    if img is None or img.size == 0:
        print("Please check image path or some error occured")
    else:
        persons_in_img = []
        img_encode = vgg_face_model.get_embeddings(img_name)
        # Make Predictions
        embed = K.eval(img_encode)
        name = classification_model.predict2(embed, 0,0,width, height, ml_data.pig_dict, img)
        persons_in_img.append(name)
        # Save images with bounding box,name and accuracy
        cv2.imwrite(os.getcwd() + '/output/recognized_img.jpg', img)
        # Pig in image
        print('Pig(s) in image is/are:' + ' '.join([str(elem) for elem in persons_in_img]))

        return img
