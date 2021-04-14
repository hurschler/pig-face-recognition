import os
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import cv2
from PIL import Image as Pil_Image
from skimage.exposure import exposure
import util.config as config
import logging.config
import glob
import numpy as np
import data_aug_varia


# based on https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
class Augmentation:
    """Horizontal shift image augmentation"""

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("init Augmentation")

    def generate_albumentation(self):
        data_aug_varia.generate_aug_images()

    def generate_sharp_img(self):
        img_path_crop = config.output_path_cropped_rectangle
        pig_img_folders = os.listdir(img_path_crop)
        for i, pig_name in enumerate(pig_img_folders):
            img_path = os.path.join(img_path_crop, pig_name)
            image_names = glob.glob(os.path.join(img_path, 'DSC*'))
            for image_name in image_names:
                image_name = os.path.basename(image_name)
                img_orig = load_img(os.path.join(img_path, image_name))
                img_orig_opencv = np.array(img_orig)
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpened = cv2.filter2D(img_orig_opencv, -1, kernel) # applying the sharpening kernel to the input image & displaying it.
                blur_img = cv2.GaussianBlur(sharpened, (0, 0), 5)
                sharpened = cv2.addWeighted(sharpened, 1.5, blur_img, -0.5, 0)
                pil_img = Pil_Image.fromarray(sharpened)
                aug_img_name = 'S-' + image_name
                pil_img.save(os.path.join(img_path, aug_img_name))
            self.log.info("augmentation in process sharpness: " + str(i))
        self.log.info('augmentation finished (sharpness)')

    def generate_contrast_img(self):
        img_path_crop = config.output_path_cropped_rectangle
        pig_img_folders = os.listdir(img_path_crop)
        for i, pig_name in enumerate(pig_img_folders):
            img_path = os.path.join(img_path_crop, pig_name)
            image_names = glob.glob(os.path.join(img_path, 'DSC*'))
            for image_name in image_names:
                image_name = os.path.basename(image_name)
                img_orig = load_img(os.path.join(img_path, image_name))
                img_orig_opencv = np.array(img_orig)
                p2, p98 = np.percentile(img_orig_opencv, (2, 98))
                img_rescale = exposure.rescale_intensity(img_orig_opencv, in_range=(p2, p98))
                pil_img = Pil_Image.fromarray(img_rescale)
                aug_img_name = 'C-' + image_name
                pil_img.save(os.path.join(img_path, aug_img_name))
            self.log.info("augmentation in process sharpness: " + str(i))
        self.log.info('augmentation finished (contrast)')

    def blur(img):
        return (cv2.blur(img,(5,5)))

    def generate_augmentation_images(self):
        img_path_crop = config.output_path_cropped_rectangle
        pig_img_folders = os.listdir(img_path_crop)
        for i, pig_name in enumerate(pig_img_folders):
            img_path = os.path.join(img_path_crop, pig_name)
            image_names = os.listdir(os.path.join(img_path_crop, pig_name))
            for image_name in image_names:
                img_orig = load_img(os.path.join(img_path, image_name))
                self.generate_augmentation(img_path, image_name, img_orig)
            self.log.info("augmentation in process: " + str(i))
        self.log.info('augmentation finished')

    def generate_augmentation(self, output_path, img_name, img):
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample

        samples = expand_dims(data, 0)
        # create image data augmentation generator

        # Augmentation Transformer
        # datagen = ImageDataGenerator(width_shift_range=[-500, 1000])
        # datagen = ImageDataGenerator(brightness_range=[0.1,0.9]),
        # datagen = ImageDataGenerator(rotation_range=45)
        # datagen=ImageDataGenerator(zoom_range=[0.2,1.6])
        # datagen=ImageDataGenerator(horizontal_flip=True)

        datagen = ImageDataGenerator(
            width_shift_range=[-100, 100],
            brightness_range=[0.4,1.1],
            rotation_range=20,
            horizontal_flip=True,
            # fill_mode='constant'
            fill_mode='nearest'
        )

        # prepare iterator
        it = datagen.flow(samples, batch_size=1)

        # generate samples and plot
        for i in range(3):
            # define subplot
            # pyplot.subplot(330 + 1 + i)
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            pil_img = Pil_Image.fromarray(image)
            aug_img_name = str(i) + '-' + img_name
            pil_img.save(os.path.join(output_path, aug_img_name))
            # plot raw pixel data
            # pyplot.imshow(image)
            # pyplot.show()
        return pyplot


# load the image
# img_name = 'DSC_V1_6460_2238.JPG'
# input_path = '../sample'
# output_path = '../output'

# img = load_img(os.path.join(input_path, img_name))
# aug = Augmentation()
# pyplot = aug.generate_augmentation(output_path, img_name, img)
# show the figure
# pyplot.show()
