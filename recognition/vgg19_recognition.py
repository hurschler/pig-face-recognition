import os
import vgg19_model
import classification_model
import vgg19_utils as vgg19_util
import ml_data
import logging.config
import util.logger_init
from keras.preprocessing import image

from recognition import classification_auto_keras_model
from util import confusion_matrix

log = logging.getLogger(__name__)
log.info("Start Nas_Net_Large_recognition")
log.info("Logger is initialized")

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# 3. Create a new VGG Face model
vgg19_model = vgg19_model.Vgg19()

# Remove Last Softmax layer and get model up to last flatten layer with outputs ???? units (transfer learning)
# vgg19_model.remove_last_layer()

# 6. Prepare Data Structures
ml_data = ml_data.MlData([],[],[],[], {})

# 7. Read Images from Disk and calculate the feature vector
# vgg19_util.calculate_feature_vectors_train(vgg19_model, ml_data)
# vgg19_util.calculate_feature_vectors_test(vgg19_model, ml_data)

# 8. convert all feature vectors to JSON File
# vgg19_util.convert_to_json_and_save(ml_data)
ml_data = vgg19_util.load_ml_data_from_json_file(ml_data)

# 9. Create a new Classification Model
classification_model = classification_model.ClassificationModel(ml_data)
# classification_model = classification_auto_keras_model.ClassificationAutoKerasModel(ml_data)

# 10. Train the Classification model with the embedding Datas
classification_model.fit(ml_data)

# 11. Export the Model
classification_model.save_model()

# Load the Model from a file
classification_model.load_model()

# img_name_full_path = r"G:\temp\pig-face-rectangle-test\6471\DSC_V1_6471_2479.JPG-crop-mask0.jpg"
# img_name_full_path = r"G:\temp\pig-face-rectangle-test\6471\DSC_V1_6471_2479.JPG-crop-mask0.jpg"
# img_name_full_path = r"G:\temp\pig-face-rectangle-test\6444\DSC_V2_6444_2568.JPG-crop-mask0.jpg"
img_name_full_path = r"G:\temp\pig-face-rectangle-test\6446\DSC_V2_6446_2774.JPG-crop-mask0.jpg"

# 12. Predict
# predict = vgg19_util.predict(vgg19_model, classification_model, ml_data, img_name_full_path)
# predict = vgg19_util.predict_validation_set(vgg19_model, classification_model, ml_data)

# confusion_matrix.create_confusion_matrix(predict, ml_data.y_test, ml_data.pig_dict, ml_data)
