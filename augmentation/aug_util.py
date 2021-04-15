import glob
import os
import logging.config
import numpy as np
from PIL import Image as Pil_Image
from matplotlib import pyplot as plt
from albumentations import *
from util import logger_init

log = logging.getLogger(__name__)


def save_aug_image(image_name, img_path, pig_img_aug1, prefix):
    """
    Saves the augmented image

    """
    log.info('Saving augmented image...')
    img_aug_opencv = np.array(pig_img_aug1)
    pil_img = Pil_Image.fromarray(img_aug_opencv)
    aug_img_name = prefix + image_name
    pil_img.save(os.path.join(img_path, aug_img_name))


def plot_image(pig_img_aug, label, rows):
    """Plots the augmented image"""
    log.info('Plotting augmented image...')
    global num
    num = num + 1
    plt.subplot(rows, 5, num)
    plt.title(label)
    plt.imshow(pig_img_aug)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Initialize the dimensions of the image to be resized and grab the
    image size.
    """
    log.info('Resizing image...')
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def clean_augmented_images(path, pattern):
    """
    Deletes all files in sub folders depending on the pattern.
    @param:
        - path : str Path of the to delete file in sub folders
        - pattern: str Pattern of the file beginning
    """
    log.info('Start deleting process of images with the pattern ' + pattern + '...')
    all_files = [y
                 for x in os.walk(path)
                 for y in glob.glob(os.path.join(x[0], pattern))
                 ]
    if len(all_files) == 0:
        log.warning('No files for this pattern ' + pattern + ' in the folders')
    for file in all_files:
        os.remove(file)
    log.info('Removed ' + str(len(all_files)) + ' Elements')
    log.info('Finished deleting process')


def counts_file_in_sub_folder_with_specific_pattern(path, pattern):
    """
       Counts all files in sub folders depending on the pattern.
       @param:
           - path : str Path of the to delete file in sub folders
           - pattern: str Pattern of the file beginning
       """
    log.info('Start counting process of images with the pattern ' + pattern + '...')
    all_files = [y
                 for x in os.walk(path)
                 for y in glob.glob(os.path.join(x[0], pattern))
                 ]
    log.info('Finished counting process - Total elements: ' + str(len(all_files)))
    return len(all_files)










