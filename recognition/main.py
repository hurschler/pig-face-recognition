import os
import ml_data
import efficientnet_model
import classification_model
import ml_util
import logging.config
import classification_auto_keras_model
import vgg_face_model
from recognition import vgg19_model, resnet_model
from util import confusion_matrix

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# NETWORK_MODEL = efficientnet_model
# feature_extraction_model = vgg19_model.Vgg19()
feature_extraction_model = efficientnet_model.EfficientNetModel()
# feature_extraction_model = inception_resnet_v2_model.InceptionResNetV2()
# feature_extraction_model = nasnetlarge_model.NasNetLarge()
# feature_extraction_model = resnet_model.ResNetModel()
# feature_extraction_model = xception_model.XceptionModel()
# feature_extraction_model = vgg_face_model.VggFaceModel()


"""Network model options are:
    - efficientnet_model
"""
CALCULATE_VECTORS = False
LOAD_WEIGHTS = True
TRAIN_WITH_AUTOKERAS = False
FIT_CLASSIFICATION_MODEL = True
PREDICT_VALIDATION_SET = False
PREDICT_SINGE_PIG_IMG = True
KFOLD_VALIDATION = True
PATH_TRAIN_DATA = ''
PATH_TEST_DATA = ''
batch_size = 10
epochs = 200
number_of_pigs = 20
k = 5


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

    ml_data = ml_util.load_ml_data_from_json_file(ml_data, feature_extraction_model)
    log.info('Creating new classification model...')
    classification_model = classification_model.ClassificationModel(ml_data, number_of_pigs)

    if TRAIN_WITH_AUTOKERAS:
        classification_model = classification_auto_keras_model.ClassificationAutoKerasModel(ml_data)

    if FIT_CLASSIFICATION_MODEL:
        log.info('Starting training of the classification model...')
        if KFOLD_VALIDATION:
            classification_model.fit_with_k_fold(ml_data, batch_size=batch_size, epochs=epochs, k=k)
        else:
            classification_model.fit(ml_data, batch_size=batch_size, epochs=epochs)
        log.info('Saving trained model...')
        classification_model.save_model()

    log.info('Loading classification model')
    classification_model.load_model()

    if PREDICT_VALIDATION_SET:
        predict = ml_util.predict_validation_set(feature_extraction_model, classification_model, ml_data)
        confusion_matrix.create_confusion_matrix(predict, ml_data.y_test, ml_data.pig_dict, ml_data)

    if PREDICT_SINGE_PIG_IMG:
        img_name_full_path = r"G:\temp\pig-face-rectangle-test\6446\DSC_V2_6446_2774.JPG-crop-mask0.jpg"
        predict = ml_util.predict(feature_extraction_model, classification_model, ml_data, img_name_full_path)
