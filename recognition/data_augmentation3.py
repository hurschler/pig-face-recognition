import glob

import numpy as np
import cv2
import os
import tensorflow as tf
from keras.preprocessing.image import load_img
from PIL import Image as Pil_Image
from matplotlib import pyplot as plt
from IPython.display import display, HTML
import util.config as config


from albumentations import *

# https://github.com/albumentations-team/albumentations#installation


def generate_aug_images():
    img_path_crop = config.output_path_cropped_rectangle
    pig_img_folders = os.listdir(img_path_crop)
    for i, pig_name in enumerate(pig_img_folders):
        img_path = os.path.join(img_path_crop, pig_name)
        image_names = glob.glob(os.path.join(img_path, 'DSC*'))
        for image_name in image_names:
            image_name = os.path.basename(image_name)
            img_orig = cv2.imread(os.path.join(img_path, image_name))
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

            alpha = 1.2
            aug = RandomBrightnessContrast(p=1)
            pig_img_aug1 = aug.apply(img_orig, alpha=alpha)
            save_aug_image(image_name, img_path, pig_img_aug1, 'A1-')

            aug = CLAHE(p=1.0)
            pig_img_aug2 = aug.apply(img_orig)
            save_aug_image(image_name, img_path, pig_img_aug2, 'A2-')

            aug = ToGray(p=0.5)
            pig_img_aug3 = aug.apply(img_orig)
            save_aug_image(image_name, img_path, pig_img_aug3, 'A3-')

            aug = Blur(p=0.5, blur_limit=7)
            pig_img_aug4 = aug.apply(img_orig)
            save_aug_image(image_name, img_path, pig_img_aug4, 'A4-')

            aug = RandomFog(p=1, fog_coef_lower=0.1, fog_coef_upper=0.1, alpha_coef=0.8)
            pig_img_aug5 = aug.apply(img_orig)
            save_aug_image(image_name, img_path, pig_img_aug5, 'A5-')

            aug = HueSaturationValue(hue_shift_limit=200, sat_shift_limit=70, val_shift_limit=27, p=1)
            pig_img_aug6 = aug.apply(img_orig)
            save_aug_image(image_name, img_path, pig_img_aug6, 'A6-')

        print("augmentation in process A1: " + str(i))
    print('augmentation finished (sharpness)')


def save_aug_image(image_name, img_path, pig_img_aug1, prefix):
    img_aug_opencv = np.array(pig_img_aug1)
    pil_img = Pil_Image.fromarray(img_aug_opencv)
    aug_img_name = prefix + image_name
    pil_img.save(os.path.join(img_path, aug_img_name))


def plot_image(pig_img_aug, label):
    global num
    num = num + 1
    plt.subplot(rows, 5, num)
    plt.title(label)
    plt.imshow(pig_img_aug)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
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
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


# pig_img = cv2.imread(r'../sample/DSC_V1_6460_2238.JPG')
# pig_img = cv2.cvtColor(pig_img, cv2.COLOR_BGR2RGB)
# pig_img = image_resize(pig_img, height=416)

# alpha = 1.2
# aug = RandomBrightnessContrast(p=1)
# pig_img_aug1 = aug.apply(pig_img, alpha=alpha)

# aug = RandomFog(p=1, fog_coef_lower=0.1, fog_coef_upper=0.1, alpha_coef=0.8)
# pig_img_aug2 = aug.apply(pig_img)

# aug = HueSaturationValue(hue_shift_limit=200, sat_shift_limit=70, val_shift_limit=27, p=1)
# pig_img_aug3 = aug.apply(pig_img)

# aug = ElasticTransform(alpha=203, sigma=25, alpha_affine=25, p=1.0)
# pig_img_aug4 = aug.apply(pig_img)

# aug = ToGray(p=0.5)
# pig_img_aug5 = aug.apply(pig_img)

# aug = CLAHE(p=1.0)
# pig_img_aug6 = aug.apply(pig_img)

# aug = Blur(p=0.5, blur_limit=7)
# pig_img_aug7 = aug.apply(pig_img)

# -----------------------------------------------------------------------------------------------------------
plt.rcParams['figure.figsize'] = [16, 8]
rows = 2
num = 0

# plot_image(pig_img, 'orig')
# plot_image(pig_img_aug1, 'brightnessContrast')
# plot_image(pig_img_aug2, 'fog')
# plot_image(pig_img_aug3, 'hueSaturation')
# plot_image(pig_img_aug4, 'elasticTransform')
# plot_image(pig_img_aug5, 'toGray')
# plot_image(pig_img_aug6, 'clahe')
# plot_image(pig_img_aug7, 'blur')

# generate_aug_images()

plt.axis('off')
plt.tight_layout()
plt.show()

cv2.waitKey(0)