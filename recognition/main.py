import os
import ml_data
import efficientnet_model
import classification_model
import ml_util
import logging.config
import classification_auto_keras_model
from recognition import vgg19_model
from util import path_switcher

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# NETWORK_MODEL = efficientnet_model
# feature_extraction_model = vgg19_model.Vgg19()
feature_extraction_model = efficientnet_model.EfficientNetModel()

"""Network model options are:
    - efficientnet_model
"""
CALCULATE_VECTORS = True
LOAD_WEIGHTS = False
TRAIN_WITH_AUTOKERAS = False
PATH_TRAIN_DATA = ''
PATH_TEST_DATA = ''
batch_size = 1
epochs = 20
number_of_pigs = 20


if __name__ == '__main__':
    """
    Main File of the pig-face-recognition project. All necessary parameters can you configure
    before the main file.
    Parameters:
    - NETWORK_MODEL: The used network for the recognition
    - CALCULATE_VECTORS: Calculates the feature vector of the chosen network with the chosen data
    - LOAD_WEIGHTS: Loads the pretrained weights of the network
    - TRAIN_WITH_AUTOKERAS: Trains the classification model with autokeras
    
    """
    log = logging.getLogger(__name__)

    if LOAD_WEIGHTS:
        log.info('Loading weights...')
        feature_extraction_model.load_weights()
    log.info('Preparing data structures...')
    ml_data = ml_data.MlData([], [], [], [], {})

    if CALCULATE_VECTORS:
        ml_util.calculate_feature_vectors_train(feature_extraction_model, ml_data)
        ml_util.calculate_feature_vectors_test(feature_extraction_model, ml_data)
        ml_util.convert_to_json_and_save(ml_data, feature_extraction_model)

    ml_data = ml_util.load_ml_data_from_json_file(ml_data)
    log.info('Creating new classification model...')
    classification_model = classification_model.ClassificationModel(ml_data)

    if TRAIN_WITH_AUTOKERAS:
        classification_model = classification_auto_keras_model.ClassificationAutoKerasModel(ml_data)

    log.info('Starting training of the classification model...')
    classification_model.fit(ml_data, batch_size=batch_size, epochs=epochs)
    log.info('Saving trained model...')
    classification_model.save_model()
    log.info('Loading classification model')
    classification_model.load_model()
