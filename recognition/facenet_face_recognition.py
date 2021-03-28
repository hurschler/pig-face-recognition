import os
import facenet_model
import classification_model
import facenet_face_recognition_utils as facenet_util
import ml_data
import logging.config
import util.logger_init
from keras.preprocessing import image


log = logging.getLogger(__name__)
log.info("Start efficientnet_face_recognition")
log.info("Logger is initialized")

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# 3. Create a new VGG Face model
facenet_face_model = facenet_model.FaceNetModel()

# 6. Prepare Data Structures
ml_data = ml_data.MlData([],[],[],[], {})

# 7. Read Images from Disk and calculate the feature vector
# facenet_util.calculate_feature_vectors_train(facenet_face_model, ml_data)
# facenet_util.calculate_feature_vectors_test(facenet_face_model, ml_data)

# 8. convert all feature vectors to JSON File
# facenet_util.convert_to_json_and_save(ml_data)
ml_data = facenet_util.load_ml_data_from_json_file(ml_data, '../output/data.json')

# 9. Create a new Classification Model
classification_model = classification_model.ClassificationModel(ml_data)

# 10. Train the Classification model with the embedding Datas
# classification_model.fit(ml_data)

# 11. Export the Model
# classification_model.save_model()

# Load the Model from a file
# efficientnet_face_model.load_model()
classification_model.load_model()

# Not ok
# img_name_full_path = r"G:\temp\pig-face-22-03-2021-test\6472\DSC_V1_6472_2270.JPG-crop-mask0.jpg"
# img_name_full_path = r"G:\temp\pig-face-22-03-2021-test\6501\DSC_V1_6501_2403.JPG-crop-mask0.jpg"
# img_name_full_path = r"G:\temp\pig-face-22-03-2021-test\6476\DSC_V1_6476_2334.JPG-crop-mask0.jpg"

# ok 0.41
img_name_full_path = r"G:\temp\pig-face-22-03-2021-test\6460\DSC_V1_6460_2247.JPG-crop-mask0.jpg"
# ok 0.10
# img_name_full_path = r"G:\temp\pig-face-22-03-2021-test\6498\DSC_V1_6498_2475.JPG-crop-mask0.jpg"

# 12. Predict
img = facenet_util.predict2(facenet_face_model, classification_model, ml_data, img_name_full_path)

