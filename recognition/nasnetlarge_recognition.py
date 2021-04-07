import os
import nasnetlarge_model
import classification_model
import nasnetlarge_utils as nas_util
import ml_data
from util.preprocessing import Preprocessing
# from recognition.data_augmentation import Augmentation
import logging.config
import util.logger_init
from keras.preprocessing import image


log = logging.getLogger(__name__)
log.info("Start Nas_Net_Large_recognition")
log.info("Logger is initialized")

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# 3. Create a new VGG Face model
nasnetlarge_model = nasnetlarge_model.NasNetLarge()

# 6. Prepare Data Structures
ml_data = ml_data.MlData([],[],[],[], {})

# 7. Read Images from Disk and calculate the feature vector
nas_util.calculate_feature_vectors_train(nasnetlarge_model, ml_data)
nas_util.calculate_feature_vectors_test(nasnetlarge_model, ml_data)

# 8. convert all feature vectors to JSON File
nas_util.convert_to_json_and_save(ml_data)
ml_data = nas_util.load_ml_data_from_json_file(ml_data)

# 9. Create a new Classification Model
classification_model = classification_model.ClassificationModel(ml_data)

# 10. Train the Classification model with the embedding Datas
# classification_model.fit(ml_data)

# 11. Export the Model
# classification_model.save_model()
# efficientnet_face_model.save_model()

# Load the Model from a file
# efficientnet_face_model.load_model()
classification_model.load_model()

# img_name_full_path = r"G:\temp\pig-face-22-03-2021-test\6472\DSC_V1_6472_2270.JPG-crop-mask0.jpg"
img_name_full_path = r"G:\temp\pig-face-rectangle-test\6471\DSC_V1_6471_2479.JPG-crop-mask0.jpg"

# 12. Predict
img = nas_util.predict2(nasnetlarge_model, classification_model, ml_data, img_name_full_path)
