import os
import glob
import aug_util
import logging
from util import config
from util import logger_init
from albumentations import ToGray, cv2
from augmentation.aug_util import save_aug_image

log = logging.getLogger(__name__)
PATH_TO_IMAGES = config.output_path_cropped_rectangle


def generate_aug_images_gray_scale(path=PATH_TO_IMAGES):
    """
    Generates augmented images in gray scale of a specific folder
    Augmentations
        - Convert the input RGB image to grayscale
    """
    log.info('Generating augmentation in gray scale of the images...')
    log.info('Path of the images: ' + PATH_TO_IMAGES)
    for i, pig_name in enumerate(os.listdir(path)):
        img_path = os.path.join(path, pig_name)
        image_names = glob.glob(os.path.join(img_path, 'DSC*'))
        for image_name in image_names:
            image_name = os.path.basename(image_name)
            img_orig = cv2.imread(os.path.join(img_path, image_name))
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

            aug = ToGray(p=0.5)
            pig_img_aug3 = aug.apply(img_orig)
            save_aug_image(image_name, img_path, pig_img_aug3, 'GS-')
        log.info('augmentation in process GS: ' + str(i))
    log.info('Augmentation is finished')

# -----------------------------------------------------------------------------------------------------------


# generate_aug_images_gray_scale()
aug_util.clean_augmented_images(path=PATH_TO_IMAGES, pattern='GS*')


