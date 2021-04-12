import os
import resnet50_model
import classification_model
import resnet50_recognition_utils as res50_util
import ml_data
import logging.config
import util.logger_init


log = logging.getLogger(__name__)
log.info("Start efficientnet_face_recognition")
log.info("Logger is initialized")

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# 1. Image Preprocessing
# pre = Preprocessing()

# 3. Create a new VGG Face model
resnet50_face_model = resnet50_model.ResNetModel()

resnet50_face_model.fit(res50_util.load_train_dataset(), res50_util.load_validate_dataset())
resnet50_face_model.save_weights()
resnet50_face_model.load_weights()

# 6. Prepare Data Structures
ml_data = ml_data.MlData([], [], [], [], {})

res50_util.calculate_feature_vectors_train(resnet50_face_model, ml_data)
res50_util.calculate_feature_vectors_test(resnet50_face_model, ml_data)

# 8. convert all feature vectors to JSON File
res50_util.convert_to_json_and_save(ml_data)
ml_data = res50_util.load_ml_data_from_json_file(ml_data)

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
