import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow.keras.backend as K
import logging.config
import util.logger_init
import util.config as config
import jsonpickle
import recognition.ml_data

log = logging.getLogger(__name__)


def convert_to_json_and_save(ml_data, path):
    """Converts the data to a json file"""
    log.info('Converting data to json...')
    ml_data_json = jsonpickle.encode(ml_data)
    log.info('Path for saving feature vector: ' + path)
    with open(path, "w") as fh:
        fh.write(ml_data_json)


def load_ml_data_from_json_file(ml_data, path):
    """Loads the Data from the json file"""
    log.info('Loading Data from json file...')
    log.info('Path for loading feature vector: ' + path)
    with open(path, "r") as fh:
        ml_data = jsonpickle.loads(fh.read())
    return ml_data


def load_train_dataset():
    """Loads the train data set"""
    log.info('Loading train data set...')
    train_datagen = ImageDataGenerator(
        horizontal_flip = True)
    train_generator = train_datagen.flow_from_directory(
        config.output_path_cropped_rectangle,
        batch_size=1,
        class_mode='binary')
    return train_generator


def load_validate_dataset():
    """Loads the validation data set"""
    log.info('Loading the validation data...')
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        config.output_path_cropped_rectangle_test,
        batch_size=1,
        class_mode='binary')
    return validation_generator


def calculate_feature_vectors_train(efficientnet_face_model, ml_data):
    """Calculates the feature vector of the train data set
    @Params:
    - class_model: specific class model of the algorithm
    - ml_data : data structure
    - path_of_train_data: Specific path of the data for training
    - target-size: Size of the input image target
    """
    log.info('Calculating feature vectors of the train data...')
    img_path_crop = config.output_path_cropped_rectangle
    pig_img_folders = os.listdir(img_path_crop)
    for i, pig_name in enumerate(pig_img_folders):
        ml_data.pig_dict[i] = pig_name
        image_names = os.listdir(os.path.join(img_path_crop, pig_name))
        for image_name in image_names:
            img = load_img(os.path.join(img_path_crop, pig_name, image_name), target_size=(600, 600))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = tf.keras.applications.efficientnet.preprocess_input(img)
            img_encode = efficientnet_face_model.efficient_net_face(img)
            feature_vector = np.squeeze(K.eval(img_encode)).tolist()
            ml_data.x_train.append(feature_vector)
            ml_data.y_train.append(i)
            print('TRAIN pig-number: ', i, ' pig_name: ', pig_name, 'image_name:  ', image_name, 'length of Feature-Vector: ', len(feature_vector), ' Feature-Vector: ', feature_vector)


def calculate_feature_vectors_test(efficientnet_face_model, ml_data):
    """Calculates the feature vector of the test data set
    @Params:
    - class_model: specific class model of the algorithm
    - ml_data : data structure
    - path_of_train_data: Specific path of the data for training
    - target-size: Size of the input image target
    """
    log.info('Calculating feature vectors of the test data...')
    img_path_crop = config.output_path_cropped_rectangle_test
    pig_img_folders = os.listdir(img_path_crop)
    for i, pig_name in enumerate(pig_img_folders):
        ml_data.pig_dict[i] = pig_name
        image_names = os.listdir(os.path.join(img_path_crop, pig_name))
        for image_name in image_names:
            img = load_img(os.path.join(img_path_crop, pig_name, image_name), target_size=(600, 600))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = tf.keras.applications.efficientnet.preprocess_input(img)
            img_encode = efficientnet_face_model.efficient_net_face(img)
            feature_vector = np.squeeze(K.eval(img_encode)).tolist()
            ml_data.x_test.append(feature_vector)
            ml_data.y_test.append(i)
            # rec_util.test_data(vgg_face_model, ml_data, i, pig_name)
            print('TEST-Vector: pig-number: ', i, ' pig_name: ', pig_name, 'image_name:  ', image_name, 'length of Feature-Vector: ', len(feature_vector), ' Feature-Vector: ', feature_vector)


def plot(img):
    """Plots a image in a certain size"""
    log.info('Plotting image...')
    plt.figure(figsize=(8, 4))
    plt.imshow(img[:, :, ::-1])
    plt.axis('off')
    plt.show()


def predict2(efficientnet_face_model, classification_model, ml_data, img_name):
    img_pil = load_img(img_name, target_size=(600, 600))
    width = 600
    height = 600

    if img_pil is None or img_pil.size == 0:
        print("Please check image path or some error occured")
    else:
        persons_in_img = []
        img_encode = efficientnet_face_model.get_embeddings(img_name)
        # Make Predictions
        print('pig_name: ', img_name, 'length of Feature-Vector: ', len(img_encode), ' Feature-Vector: ', img_encode)
        name = classification_model.predict2(img_encode, 0,0,width, height, ml_data.pig_dict, img_pil)
        persons_in_img.append(name)
        # Save images with bounding box,name and accuracy
        img_opencv = np.array(img_pil)
        img_opencv = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
        cv2.imwrite('../output/recognized_img.jpg', np.array(img_opencv))
        # Pig in image
        print('Pig(s) in image is/are:' + ' '.join([str(elem) for elem in persons_in_img]))
        return img_opencv
