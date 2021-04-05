# https://github.com/ayoolaolafenwa/PixelLib/
import os
import glob
from pixellib.custom_train import instance_custom_training
from pixellib.instance import custom_segmentation
import util.config as project_config
from datetime import datetime
import time
import logging.config


class pigDetection:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def __init__(self):
        print(self)
        self.log = logging.getLogger(__name__)
        self.log.info('Init Detection')

    def train_model(self):
        train_maskrcnn = instance_custom_training()
        train_maskrcnn.modelConfig(network_backbone="resnet101", num_classes=1, batch_size=5)
        train_maskrcnn.load_pretrained_model("../model/mask_rcnn_model.036-0.139239.h5")
        self.log.info('Starting training model detection...')
        train_maskrcnn.load_dataset("images")
        train_maskrcnn.train_model(num_epochs=100, augmentation=True,  path_trained_models="model")
        train_maskrcnn.evaluate_model("model/mask_rcnn_model.003-0.700727.h5")
        self.log.info('Finished training')

    def segmenting_images(self):
        self.log.info('Loading model...')
        segment_image = custom_segmentation()
        segment_image.inferConfig(num_classes=1, class_names=["PigFace"], detection_threshold=0.95)
        # segment_image.load_model("model/model.h5")
        segment_image.load_model("../model/mask_rcnn_model.036-0.139239.h5")

        # segment_image.segmentImage("1.png", "../app/upload/1.png" , show_bboxes=True, output_image_name="../output/1.png", verbose=True)

        dir_path = project_config.image_upload_dir_path
        output_path = project_config.output_dir_path
        while True:
            self.log.info('Ready to upload directory', datetime.now())
            i = 0
            files = [x for x in os.listdir(dir_path) if (x.endswith('.jpg') or x.endswith('.JPG') or x.endswith('.png') or x.endswith('.PNG'))]

            for imageFullFileName in files:
                image_file_name = os.path.basename(imageFullFileName)
                self.log.info('Image filename: ', image_file_name)
                start_time = datetime.now()
                try:
                    segment_image.segmentImage(image_file_name, dir_path + r"/" + image_file_name, show_bboxes=True, output_image_name=output_path + r"/" + image_file_name)
                except:
                    self.log.info('Image filename: ', image_file_name)
                os.remove(os.path.join(dir_path, imageFullFileName))
                end_time = datetime.now()
                diff = (end_time-start_time).microseconds / 1000
                self.log.info('Elapsed time for Segmentation: ', '%.2gs' % diff)
            time.sleep(0.2)