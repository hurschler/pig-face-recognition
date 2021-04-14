import os
import vgg_face_model
import classification_model
import vgg_face_recognition_utils_v1 as rec_util
import ml_data
from util.preprocessing import Preprocessing
from augmentation import Augmentation
import logging.config

log = logging.getLogger(__name__)
log.info("Start vgg_face_recognition_v1")
log.info("Logger is initialized")

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# 1. Image Preprocessing
pre = Preprocessing()

# 2. Image Augmentation
aug = Augmentation()
# aug.clean_augmented_images()
# aug.generate_augmentation_images()
# aug.generate_sharp_img()

# 3. Create a new VGG Face model
vgg_face_model = vgg_face_model.VggFaceModel()

# 4. Load Weights for the VGG Model
vgg_face_model.load_weights()

# 5. Remove Last Softmax layer and get model up to last flatten layer with outputs 2622 units (transfer learning)
vgg_face_model.remove_last_layer()

# 6. Prepare Data Structures
ml_data = ml_data.MlData([],[],[],[], {})

# 7. Read Images from Disk and calculate the feature vector
rec_util.calculate_feature_vectors_train(vgg_face_model, ml_data)
rec_util.calculate_feature_vectors_test(vgg_face_model, ml_data)

# 8. convert all feature vectors to JSON File
rec_util.convert_to_json_and_save(ml_data)

# 9. Create a new Classification Model
classification_model = classification_model.ClassificationModel(ml_data)

# 10. Train the Classification model with the embedding Datas
classification_model.fit(ml_data)

# 11. Export the Model
classification_model.save_model()

# 12. Predict
# img_name_full_path = r"G:\temp\pig-face-rectangle-test\6446\DSC_V2_6446_2774.JPG-crop-mask0.jpg"
# img_name_full_path = r"G:\temp\pig-face-rectangle-test\6446\DSC_V2_6446_2775.JPG-crop-mask0.jpg"

# img = rec_util.predict2(vgg_face_model, classification_model, ml_data, img_name_full_path)

