import os
import cv2
import glob
import numpy as np
import json
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

FEATURE_VECTOR_PATH = '../feature_vector/'


def convert_to_json_and_save(ml_data, feature_extraction_model):
    """Converts the data to a json file"""
    log.info('Converting data to json...')
    feature_vector_name = feature_extraction_model.get_feature_vector_name()
    path = os.path.join(FEATURE_VECTOR_PATH, feature_vector_name)
    ml_data_json = jsonpickle.encode(ml_data)
    log.info('Path for saving feature vector: ' + path)
    with open(path, "w") as fh:
        fh.write(ml_data_json)


def load_ml_data_from_json_file(ml_data, feature_extraction_model):
    """Loads the Data from the json file"""
    log.info('Loading Data from json file...')
    feature_vector_name = feature_extraction_model.get_feature_vector_name()
    path = os.path.join(FEATURE_VECTOR_PATH, feature_vector_name)
    log.info('Path for loading feature vector: ' + path)
    with open(path, "r") as fh:
        ml_data = jsonpickle.loads(fh.read())
    return ml_data


def load_train_dataset():
    """Loads the train data set"""
    log.info('Loading train data set...')
    train_datagen = ImageDataGenerator(
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        config.output_path_cropped_rectangle,
        batch_size=1,
        class_mode='binary')
    return train_generator


def load_validate_dataset():
    """Loads the validation data set"""
    log.info('Loading the validation data...')
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(
        config.output_path_cropped_rectangle_test,
        batch_size=1,
        class_mode='binary')
    return validation_generator

def check_is_augmented_image_name(image_name):
    return not image_name.startswith("DSC")

def calculate_feature_vectors_train(feature_extractor_model, ml_data):
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
    img_index = 0
    orig_img_index = 0
    for i, pig_name in enumerate(pig_img_folders):
        ml_data.pig_dict[i] = pig_name
        image_names = os.listdir(os.path.join(img_path_crop, pig_name))
        for image_name in image_names:
            img = load_img(os.path.join(img_path_crop, pig_name, image_name),
                           target_size=feature_extractor_model.get_target_size())
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = feature_extractor_model.preprocessing_input(img)
            img_encode = feature_extractor_model.get_embeddings(img)
            feature_vector = np.squeeze(K.eval(img_encode)).tolist()
            ml_data.x_train.append(feature_vector)
            ml_data.y_train.append(i)

            ml_data.index_to_image_name[img_index] = image_name
            if check_is_augmented_image_name(image_name):
                orig_image_name = image_name.partition("-")[2]
                ml_data.image_name_to_orig_image[image_name] = orig_image_name
            else:
                ml_data.orig_x_train.append(feature_vector)
                ml_data.orig_y_train.append(i)
                ml_data.orig_index[orig_img_index] = img_index
                ml_data.image_name_to_orig_image[image_name] = image_name
                orig_img_index = orig_img_index + 1
            img_index = img_index + 1
            print('TRAIN pig-number: ', i, ' pig_name: ', pig_name, 'image_name:  ', image_name,
                  'length of Feature-Vector: ', len(feature_vector), ' Feature-Vector: ', feature_vector)


def calculate_feature_vectors_test(feature_extractor_model, ml_data):
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
    img_index = ml_data.last_key_value_of(ml_data.index_to_image_name)
    img_index = img_index + 1
    orig_img_index = ml_data.last_key_value_of(ml_data.orig_index)
    orig_img_index = orig_img_index + 1
    for i, pig_name in enumerate(pig_img_folders):
        ml_data.pig_dict[i] = pig_name
        image_names = os.listdir(os.path.join(img_path_crop, pig_name))
        for image_name in image_names:
            img = load_img(os.path.join(img_path_crop, pig_name, image_name),
                           target_size=feature_extractor_model.get_target_size())
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = feature_extractor_model.preprocessing_input(img)
            img_encode = feature_extractor_model.get_embeddings(img)
            feature_vector = np.squeeze(K.eval(img_encode)).tolist()
            ml_data.x_test.append(feature_vector)
            ml_data.y_test.append(i)

            ml_data.index_to_image_name[img_index] = image_name
            if check_is_augmented_image_name(image_name):
                orig_image_name = image_name.partition("-")[2]
                ml_data.image_name_to_orig_image[image_name] = orig_image_name
            else:
                ml_data.orig_x_test.append(feature_vector)
                ml_data.orig_y_test.append(i)
                ml_data.orig_index[orig_img_index] = img_index
                ml_data.image_name_to_orig_image[image_name] = image_name
                orig_img_index = orig_img_index + 1
            img_index = img_index + 1
            print('TEST-Vector: pig-number: ', i, ' pig_name: ', pig_name, 'image_name:  ', image_name,
                  'length of Feature-Vector: ', len(feature_vector), ' Feature-Vector: ', feature_vector)


def calculate_single_feature_vectors(feature_extractor_model, new_pig_path, img_name):
    log.info('Calculating a single feature vectors...')
    img = load_img(os.path.join(new_pig_path, img_name), target_size=feature_extractor_model.get_target_size())
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = feature_extractor_model.preprocessing_input(img)
    img_encode = feature_extractor_model.get_embeddings(img)
    feature_vector = np.squeeze(K.eval(img_encode)).tolist()
    print('image_path:  ', new_pig_path,
          'length of Feature-Vector: ', len(feature_vector), ' Feature-Vector: ', feature_vector)
    return feature_vector


def add_new_pig_to_feature_vector_set(feature_extractor_model, pig_name, number_of_pigs, ml_data):
    log.info('add feature-vectors of the new pig to the training set')
    # new_pig_path_train = config.image_new_pig_path_train
    new_pig_path = '../input'
    new_pig_path_train = os.path.join(new_pig_path, str(pig_name), 'train')
    image_names = os.listdir(new_pig_path_train)
    for image_name in image_names:
        feature_vector = calculate_single_feature_vectors(feature_extractor_model, new_pig_path_train, image_name)
        ml_data.x_train.append(feature_vector)
        ml_data.y_train.append(number_of_pigs - 1)
    ml_data.pig_dict[number_of_pigs - 1] = pig_name

    # image_new_pig_path_validation = config.image_new_pig_path_validation
    image_new_pig_path_validation = os.path.join(new_pig_path, str(pig_name), 'test')
    image_names = os.listdir(image_new_pig_path_validation)
    for image_name in image_names:
        feature_vector = calculate_single_feature_vectors(feature_extractor_model,
                                                          image_new_pig_path_validation, image_name)
        ml_data.x_test.append(feature_vector)
        ml_data.y_test.append(number_of_pigs - 1)

    return ml_data


def plot(img):
    """Plots a image in a certain size"""
    log.info('Plotting image...')
    plt.figure(figsize=(8, 4))
    plt.imshow(img[:, :, ::-1])
    plt.axis('off')
    plt.show()


def predict(feature_extraction_model, classification_model, ml_data, img_name):
    persons_in_img = []
    img = load_img(img_name, target_size=feature_extraction_model.get_target_size())
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = feature_extraction_model.preprocessing_input(img)
    width, height = feature_extraction_model.get_target_size()
    img_encode = feature_extraction_model.get_embeddings(img)
    log.info('pig_name: ' + img_name + 'length of Feature-Vector: ' + str(len(img_encode)) + ' Feature-Vector: ' + str(
        img_encode))
    name, acc = classification_model.predict(img_encode, 0, 0, width, height, ml_data.pig_dict, img)
    persons_in_img.append(name)
    # Save images with bounding box,name and accuracy
    img_opencv = np.array(img)
    img_opencv = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
    cv2.imwrite('../output/recognized_img.jpg', np.array(img_opencv))
    # Pig in image
    log.info('Pig(s) in image is/are:' + ' '.join([str(elem) for elem in persons_in_img]))
    return name, acc


def predict_validation_set(feature_extraction_model, classification_model, ml_data):
    log.info("readImages names with sub dir")
    dir_path = config.output_path_cropped_rectangle_test
    log.info("image_dir: " + dir_path)
    files = glob.glob(dir_path + r"/*/")
    y_pred = []
    i = 0
    for path_element in files:
        if os.path.isdir(path_element):
            log.debug(path_element)
            files_on_sub_dir = glob.glob(os.path.join(dir_path, path_element) + r"\*.JPG")
            for image_on_sub_dir in files_on_sub_dir:
                log.info(str(i) + " image on sub dir: " + image_on_sub_dir)
                label_nr = predict_label(feature_extraction_model, classification_model, ml_data, image_on_sub_dir)
                if label_nr in ml_data.pig_dict.keys():
                    print('Key found')
                    name = ml_data.pig_dict[label_nr]
                else:
                    print('Key not found, try with string type')
                    name = ml_data.pig_dict[str(label_nr)]
                y_pred.append(name)
                i = i + 1

    return y_pred


def predict_label(feature_extraction_model, classification_model, ml_data, img_name):
    target_size = feature_extraction_model.get_target_size()
    img_pil = load_img(img_name, target_size=target_size)
    if img_pil is None or img_pil.size == 0:
        print("Please check image path or some error occured")
    else:
        img = img_to_array(img_pil)
        img = np.expand_dims(img, axis=0)
        img = feature_extraction_model.preprocessing_input(img)
        img_encode = feature_extraction_model.get_embeddings(img)
        label_nr = classification_model.predict_label(img_encode)

    return label_nr

def export_result_to_json(uuid, pig_name, accuracy):
    result = json.dumps({"imageId": uuid, "pig_name": str(pig_name), "accuracy": str(accuracy)})
    log.info(result)
    path = config.output_dir_path
    path = os.path.join(path, 'result-' + uuid + '.json')
    with open(path, "w") as fh:
        fh.write(result)

