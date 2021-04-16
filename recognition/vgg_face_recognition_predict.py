import os
import vgg_face_model
import classification_model
import classification_auto_keras_model
import vgg_face_recognition_utils_v1 as rec_util
import ml_data
from util.preprocessing import Preprocessing
import logging.config
import util.logger_init


log = logging.getLogger(__name__)
log.info("Logger is initialized")
log.info("Start predection")

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

PATH_JSON_DATA = '../output/data-vgg16.json'


# Create a new VGG Face model
vgg_face_model = vgg_face_model.VggFaceModel()
# Load Weights for the VGG Model
# vgg_face_model.load_weights()
# Remove Last Softmax layer and get model up to last flatten layer with outputs 2622 units (transfer learning)
# vgg_face_model.remove_last_layer()
# Prepare Data Structures
ml_data = ml_data.MlData([],[],[],[], {})
ml_data = rec_util.load_ml_data_from_json_file(ml_data, json_data=PATH_JSON_DATA)
# Create a new Classification Model
classification_model = classification_model.ClassificationModel(ml_data)
# classification_model = classification_auto_keras_model.ClassificationAutoKerasModel(ml_data)
# Train the Classification model with the embedding Datas
classification_model.fit(ml_data)
# Export the Model
classification_model.save_model()
# Load the Model from a file
classification_model.load_model()
# Predict

# img_name_full_path = r"G:\temp\pig-face-rectangle-test\6460\DSC_V1_6460_2239.JPG-crop-mask0.jpg"
# img_name_full_path = r"G:\temp\pig-face-rectangle-test\6385\DSC_V2_6385_2622.JPG-crop-mask0.jpg"


# img = rec_util.predict2(vgg_face_model, classification_model, ml_data, img_name_full_path)
# Visualize debug informations
# vgg_face_model.debug_model(r"D:\Users\avatar\OneDrive - Hochschule Luzern\bearbeitet_mit_label\train\DSC_V1_6460_2238.JPG")

# vgg_face_model.debug_model(img_name_full_path)
# Visualize the result
# rec_util.plot(img)

