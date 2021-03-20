# https://github.com/ayoolaolafenwa/PixelLib/
import os
import glob
from pixellib.custom_train import instance_custom_training
from pixellib.instance import custom_segmentation
# import detection.config as project_config
import util.detection_config as project_config
from datetime import datetime
import time

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

train_maskrcnn = instance_custom_training()
train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes=1, batch_size=5)
train_maskrcnn.load_pretrained_model("../model/mask_rcnn_model.036-0.139239.h5")

# Model Training
# train_maskrcnn.load_dataset("images")
# train_maskrcnn.train_model(num_epochs=100, augmentation=True,  path_trained_models="model")
# train_maskrcnn.evaluate_model("model/mask_rcnn_model.003-0.700727.h5")
# print("finish training")

print("load ML Model")
segment_image = custom_segmentation()
segment_image.inferConfig(num_classes=1, class_names=["PigFace"], detection_threshold=0.95)
# segment_image.load_model("model/model.h5")
segment_image.load_model("../model/mask_rcnn_model.036-0.139239.h5")

# segment_image.segmentImage("1.png", "../app/upload/1.png" , show_bboxes=True, output_image_name="../output/1.png", verbose=True)

dir_path = project_config.image_upload_dir_path
output_path = project_config.output_dir_path
while True:
    print("read upload directory: ", datetime.now())
    i = 0
    files = [x for x in os.listdir(dir_path) if x.endswith('.jpg')]

    for imageFullFileName in files:
        image_file_name = os.path.basename(imageFullFileName)
        print ("Image Filename: ", image_file_name)
        start_time = datetime.now()
        try:
            segment_image.segmentImage(image_file_name, dir_path + r"/" + image_file_name, show_bboxes=True, output_image_name=output_path + r"/" + image_file_name)
        except:
            print("error on Image:" + image_file_name)
        os.remove(os.path.join(dir_path, imageFullFileName))
        end_time = datetime.now()
        diff = (end_time-start_time).microseconds / 1000
        print("Elapsedtime for Segmentation: ", "%.2gs" % diff)
    time.sleep(0.2) # Delay for 1 minute (60 seconds).

