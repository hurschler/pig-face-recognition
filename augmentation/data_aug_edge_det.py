import glob

import mpimg as mpimg
import numpy as np
import cv2
import os
import scipy.misc as sm
import tensorflow as tf
from keras.preprocessing.image import load_img
from PIL import Image as Pil_Image
from matplotlib import pyplot as plt
from IPython.display import display, HTML
from scipy import ndimage
from skimage.color import rgb2gray

import util.config as config
from albumentations import *
import logging.config
from skimage import exposure
from matplotlib import image
import util.logger_init


log = logging.getLogger(__name__)


# https://github.com/albumentations-team/albumentations#installation
def generate_aug_images():
    img_path_crop = '/Users/patrickrichner/Desktop/FH/OneDrive - Hochschule Luzern/BDA2021/07_Daten/small_dataset/test/train'
    pig_img_folders = os.listdir(img_path_crop)
    for i, pig_name in enumerate(pig_img_folders):
        img_path = os.path.join(img_path_crop, pig_name)
        image_names = glob.glob(os.path.join(img_path, 'DSC*'))
        for image_name in image_names:
            image_name = os.path.basename(image_name)
            img_keras = load_img(os.path.join(img_path, image_name))
            img_np = np.array(img_keras)
            edges = cv2.Canny(img_np, 100, 200, 3)
            plt.subplot(121), plt.imshow(img_np, cmap='gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(edges, cmap='gray')
            plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
            plt.show()
            save_aug_image(image_name, img_path, img_np, 'E-')
            log.info('Augmentation in process Edge:' + str(i))
    log.info('Augmentation finished ')


def sobel_filters():
    img_path = '/sample'
    image_name = 'DSC_V1_6460_2238.JPG'
    img_keras = load_img(os.path.join(img_path, image_name))
    img = np.array(img_keras)

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    plt.imshow(theta)
    plt.show()
    return (G, theta)


def save_aug_image(image_name, img_path, pig_img_aug1, prefix):
    log.info('Saving image...')
    img_aug_opencv = np.array(pig_img_aug1)
    pil_img = Pil_Image.fromarray(img_aug_opencv)
    aug_img_name = prefix + image_name
    pil_img.save(os.path.join(img_path, aug_img_name))

    def rgb2gray(rgb):

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray

def load_data(dir_name='faces_imgs'):
    '''
    Load images from the "faces_imgs" directory
    Images are in JPG and we convert it to gray scale images
    '''
    imgs = []
    for filename in os.listdir(dir_name):
        if os.path.isfile(dir_name + '/' + filename):
            img = mpimg.imread(dir_name + '/' + filename)
            img = rgb2gray(img)
            imgs.append(img)
    return imgs

def visualize(imgs, format=None, gray=False):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        plt_idx = i + 1
        plt.subplot(2, 2, plt_idx)
        plt.imshow(img, format)
    plt.show()


# generate_aug_images()
sobel_filters()


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

# cv2.waitKey(0)