import os
import vgg_face_model
import classification_model
import vgg_face_recognition_utils_v1 as rec_util
import ml_data
from util.preprocessing import Preprocessing
from recognition.data_augmentation import Augmentation
import logging.config
import util.logger_init


log = logging.getLogger(__name__)
log.info("Start vgg_face_recognition_v1")
log.info("Logger is initialized")

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# read Images for preprocessing
# pre.readImage()
pre = Preprocessing()
# img_dic = pre.readImages()
# img_dic = pre.readImagesWithSubDir()
# img_dic = pre.toGrey(img_dic)
# img_dic = pre.scale(img_dic)
# pre.show(img_dic)
# img_dic = pre.toEqualizeHist(img_dic)


# Image Preprocessing

# Image Augmentation
aug = Augmentation()
aug.generate_augmentation_images()
# Create a new VGG Face model
vgg_face_model = vgg_face_model.VggFaceModel()
# Load Weights for the VGG Model
vgg_face_model.load_weights()
# Remove Last Softmax layer and get model up to last flatten layer with outputs 2622 units (transfer learning)
vgg_face_model.remove_last_layer()
# Prepare Data Structures
ml_data = ml_data.MlData([],[],[],[], {})
# Read Images from Disk and calculate the feature vector
rec_util.calculate_feature_vectors(vgg_face_model, ml_data)
# convert all feature vectors to JSON File
rec_util.convert_to_json_and_save(ml_data)
# Create a new Classification Model
classification_model = classification_model.ClassificationModel(ml_data)
# Train the Classification model with the embedding Datas
classification_model.fit(ml_data)
# Export the Model
classification_model.save_model()
# Predict
# img = rec_util.predict2(vgg_face_model, classification_model, ml_data, "../image_input/DSC_V1_6462_2320.JPG")
# img = rec_util.predict2(vgg_face_model, classification_model, ml_data, r"D:\Users\avatar\OneDrive - Hochschule Luzern\bearbeitet_mit_label\train\DSC_V1_6460_2238.JPG")
img = rec_util.predict2(vgg_face_model, classification_model, ml_data, r"G:\temp\pig-face-22-03-2021\6460\DSC_V1_6460_2238.JPG-crop-mask0.jpg")

# img = rec_util.predict(vgg_face_model, classification_model, ml_data, "putin-patrick.jpg")
# Visualize debug informations
vgg_face_model.debug_model(r"D:\Users\avatar\OneDrive - Hochschule Luzern\bearbeitet_mit_label\train\DSC_V1_6460_2238.JPG")
# Visualize the result
rec_util.plot(img)

