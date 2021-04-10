import os
import numpy as np
import resnet_model
import classification_model
import resnet_face_recognition_utils as res_util
import ml_data
from recognition import classification_auto_keras_model
from util.preprocessing import Preprocessing
from recognition.data_augmentation import Augmentation
import logging.config
import util.logger_init
from keras.preprocessing import image


log = logging.getLogger(__name__)
log.info("Start efficientnet_face_recognition")
log.info("Logger is initialized")

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# 1. Image Preprocessing
# pre = Preprocessing()

# 2. Image Augmentation
# aug = Augmentation()
# aug.generate_augmentation_images()
# aug.generate_sharp_img()

# 3. Create a new VGG Face model
resnet_face_model = resnet_model.ResNetModel()

resnet_face_model.fit(res_util.load_train_dataset(), res_util.load_validate_dataset())
resnet_face_model.save_weights()
resnet_face_model.load_weights()

# 6. Prepare Data Structures
ml_data = ml_data.MlData([], [], [], [], {})


res_util.calculate_feature_vectors_train(resnet_face_model, ml_data)
res_util.calculate_feature_vectors_test(resnet_face_model, ml_data)


# 8. convert all feature vectors to JSON File
res_util.convert_to_json_and_save(ml_data)
ml_data = res_util.load_ml_data_from_json_file(ml_data)

# 9. Create a new Classification Model
classification_model = classification_model.ClassificationModel(ml_data)
# classification_model = classification_auto_keras_model.ClassificationAutoKerasModel(ml_data)

# 10. Train the Classification model with the embedding Datas
classification_model.fit(ml_data)

# 11. Export the Model
classification_model.save_model()
# efficientnet_face_model.save_model()

# Load the Model from a file
# efficientnet_face_model.load_model()
classification_model.load_model()


# 12. Predict
# img = res_util.predict2(resnet_face_model, classification_model, ml_data, img_name_full_path)
# img = eff_util.predict2(vgg_face_model, classification_model, ml_data, r"/Users/patrickrichner/Desktop/FH/11.Semester/Bda2021/pig-face-recognition/data/validate/6357/DSC_V2_6357_2762.JPG-crop-mask0.jpg")
# Visualize debug informations
# vgg_face_model.debug_model(r"D:\Users\avatar\OneDrive - Hochschule Luzern\bearbeitet_mit_label\train\DSC_V1_6460_2238.JPG")
# Visualize the result
# rec_util.plot(img)

