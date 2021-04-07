import os
import numpy as np
# import vgg_face_model
import efficientnet_model
import classification_model
import efficientnet_face_recognition_utils as eff_util
import ml_data
from recognition import classification_auto_keras_model
from util.preprocessing import Preprocessing
from recognition.data_augmentation import Augmentation
import logging.config
import util.logger_init
from keras.preprocessing import image


#log = logging.getLogger(__name__)
#log.info("Start efficientnet_face_recognition")
#log.info("Logger is initialized")

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# 1. Image Preprocessing
# pre = Preprocessing()

# 2. Image Augmentation
# aug = Augmentation()
# aug.generate_augmentation_images()
# aug.generate_sharp_img()

# 3. Create a new VGG Face model
efficientnet_face_model = efficientnet_model.EfficientNetModel()

# 4. Load Weights for the VGG Model
# efficientnet_face_model.load_weights()

# 5. Remove Last Softmax layer and get model up to last flatten layer with outputs 2622 units (transfer learning)
# vgg_face_model.remove_last_layer()

# 6. Prepare Data Structures
ml_data = ml_data.MlData([],[],[],[], {})

# 7. Read Images from Disk and calculate the feature vector
# train_generator = eff_util.load_train_dataset()
# validation_generator = eff_util.load_validate_dataset()

# efficientnet_face_model.fit(train_generator, validation_generator)
# efficientnet_face_model.fit(ml_data)

# eff_util.calculate_feature_vectors_train(efficientnet_face_model, ml_data)
# eff_util.calculate_feature_vectors_test(efficientnet_face_model, ml_data)


# 8. convert all feature vectors to JSON File
# eff_util.convert_to_json_and_save(ml_data)
ml_data = eff_util.load_ml_data_from_json_file(ml_data)

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

# img_name_full_path = r"G:\temp\pig-face-22-03-2021-test\6472\DSC_V1_6472_2270.JPG-crop-mask0.jpg"

# img_name_full_path = r"G:\temp\pig-face-22-03-2021-test\6501\DSC_V1_6501_2403.JPG-crop-mask0.jpg"

# ok 0.41
# img_name_full_path = r"G:\temp\pig-face-22-03-2021-test\6460\DSC_V1_6460_2247.JPG-crop-mask0.jpg"
# img_name_full_path = r"G:\temp\pig-face-22-03-2021-test\6498\DSC_V1_6498_2475.JPG-crop-mask0.jpg"

# img = image.load_img(img_name_full_path, target_size=(224, 224))
# img_tensor = image.img_to_array(img)                    # (height, width, channels)
# img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
# img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
# check prediction
# pred = efficientnet_face_model.getModel().predict(img_tensor, verbose=1)

# print ('prediction Schweinchen Index: ' + str(np.argmax(pred)))

# 12. Predict
img = eff_util.predict2(efficientnet_face_model, classification_model, ml_data, img_name_full_path)
# img = eff_util.predict2(vgg_face_model, classification_model, ml_data, r"/Users/patrickrichner/Desktop/FH/11.Semester/Bda2021/pig-face-recognition/data/validate/6357/DSC_V2_6357_2762.JPG-crop-mask0.jpg")
# Visualize debug informations
# vgg_face_model.debug_model(r"D:\Users\avatar\OneDrive - Hochschule Luzern\bearbeitet_mit_label\train\DSC_V1_6460_2238.JPG")
# Visualize the result
# rec_util.plot(img)

