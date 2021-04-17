import ml_data
import efficientnet_model
import classification_model
import ml_util
import logging.config
import classification_auto_keras_model
import path_switcher


NETWORK_MODEL = efficientnet_model
"""Network model options are:
    - efficientnet_model
"""
CALCULATE_VECTORS = True
LOAD_WEIGHTS = False
TRAIN_WITH_AUTOKERAS = False
PATH_FEATURE_VECTOR = ''
PATH_TRAIN_DATA = ''
PATH_TEST_DATA = ''
TARGET_SIZE = (224, 224)
BATCH_SIZE = 1
EPOCHS = 20


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
    path_switcher = path_switcher.PathSwitcher()

    def case_1():
        global PATH_FEATURE_VECTOR
        PATH_FEATURE_VECTOR = '../feature_vector/efficient-net-b7.json'
        log.info('Path for feature vector: ' + PATH_FEATURE_VECTOR)
        return PATH_FEATURE_VECTOR

    path_switcher.add_case(efficientnet_model, case_1, True)
    path_switcher.case(NETWORK_MODEL)

    log.info('Creating new network model...')
    network_model = NETWORK_MODEL.EfficientNetModel()

    if LOAD_WEIGHTS:
        log.info('Loading weights...')
        network_model.load_weights()
    log.info('Preparing data structures...')
    ml_data = ml_data.MlData([], [], [], [], {})

    if CALCULATE_VECTORS:
        ml_util.calculate_feature_vectors_train(network_model, ml_data)
        ml_util.calculate_feature_vectors_test(network_model, ml_data)
        ml_util.convert_to_json_and_save(ml_data, path=PATH_FEATURE_VECTOR)

    ml_data = ml_util.load_ml_data_from_json_file(ml_data, path=PATH_FEATURE_VECTOR)
    log.info('Creating new classification model...')
    classification_model = classification_model.ClassificationModel(ml_data)

    if TRAIN_WITH_AUTOKERAS:
        classification_model = classification_auto_keras_model.ClassificationAutoKerasModel(ml_data)

    log.info('Starting training of the classification model...')
    classification_model.fit(ml_data, batch_size=BATCH_SIZE, epochs=EPOCHS)
    log.info('Saving trained model...')
    classification_model.save_model()
    log.info('Loading classification model')
    classification_model.load_model()
