# https://github.com/ayoolaolafenwa/PixelLib/
import os
import glob
from pixellib.custom_train import instance_custom_training
from pixellib.instance import custom_segmentation
# import detection.config as project_config
import util.detection_config as project_config

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

train_maskrcnn = instance_custom_training()
train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes=1, batch_size=5)
train_maskrcnn.load_pretrained_model("../model/mask_rcnn_coco.h5")

# Model Training
# train_maskrcnn.load_dataset("images")
# train_maskrcnn.train_model(num_epochs=100, augmentation=True,  path_trained_models="model")
# train_maskrcnn.evaluate_model("model/mask_rcnn_model.003-0.700727.h5")
# print("finish training")

print("start segemntation")
segment_image = custom_segmentation()
segment_image.inferConfig(num_classes=1, class_names=["PigFace"], detection_threshold=0.95)
# segment_image.load_model("model/model.h5")
segment_image.load_model("../model/mask_rcnn_model.006-0.181393.h5")

segment_image.segmentImage("1.png", "../app/upload/1.png" , show_bboxes=True, output_image_name="../output/1.png", verbose=True)

# dir_path = project_config.image_train_dir_path
# output_path = project_config.output_path_cropped
# i = 0
# files = glob.glob(dir_path + r"\*.JPG")
# max_image_number = 10
# for imageFullFileName in files:
#     if i >= max_image_number:
#         break
#     image_file_name = os.path.basename(imageFullFileName)
#     print ("Image Filename: ", image_file_name)
#    segment_image.segmentImage(image_file_name, dir_path + r"/" + image_file_name, show_bboxes=True, output_image_name=output_path + r"/" + image_file_name + "-mask_rcnn_model.006-0.181393.png", verbose=True)

