import os
import recognition.efficientnet_model
import ml_util
import logging.config
import shutil
import time

from ml_data import MlData
from efficientnet_model import EfficientNetModel
from classification_model import ClassificationModel

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

feature_extraction_model = EfficientNetModel()

PATH_TRAIN_DATA = ''
PATH_TEST_DATA = ''
batch_size = 10
epochs = 100
# pig_name = 6950
log = logging.getLogger(__name__)

def add_new_pig(pig_name):
    log.info('Loading weights...')
    feature_extraction_model.load_weights()
    log.info('Preparing data structures...')
    ml_data = MlData([], [], [], [], {})
    ml_data = ml_util.load_ml_data_from_json_file(ml_data, feature_extraction_model)
    log.info('ml_data size of trainingset bevor new pig: ' + str(len(ml_data.x_train)))
    number_of_pigs = len(ml_data.pig_dict) + 1
    ml_data = ml_util.add_new_pig_to_feature_vector_set(feature_extraction_model, pig_name, number_of_pigs, ml_data)
    log.info('ml_data size of trainingset after new pig: ' + str(len(ml_data.x_train)))
    log.info('Creating new classification model...')
    classification_model = ClassificationModel(ml_data, number_of_pigs)
    log.info('Loading classification model')
    # classification_model.load_model()
    classification_model.fit(ml_data, batch_size=batch_size, epochs=epochs)
    log.info('Saving trained model...')
    classification_model.save_model()


i = 0
while True:
    log.info('Ready input directory...')
    input_path = '../input'
    pig_name_dir_list = os.listdir(input_path)
    for dir_name in pig_name_dir_list:
        log.info('New Pig Name: ' + dir_name)
        new_pig_dir_name = os.path.join(input_path, dir_name)
        if (os.path.isdir(os.path.join(input_path, dir_name))):
            files = [x for x in os.listdir(os.path.join(input_path, dir_name)) if (x == 'finished.txt')]
            if (len(files) > 0):
                add_new_pig(dir_name)
                shutil.rmtree(new_pig_dir_name)
    i = i + 1
    print ('read input dir again... ' + str(i))
    time.sleep(1)


