# example of horizontal shift image augmentation
import os
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from PIL import Image as Pil_Image
from util.preprocessing import Preprocessing
import util.detection_config as detection_config

from matplotlib import pyplot as plt


# based on https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/

class Augmentation:

    def generate_augmentation_images(self):
        img_path_crop = detection_config.output_path_cropped_rectangle
        pig_img_folders = os.listdir(img_path_crop)
        for i, pig_name in enumerate(pig_img_folders):
            img_path = os.path.join(img_path_crop, pig_name)
            image_names = os.listdir(os.path.join(img_path_crop, pig_name))
            for image_name in image_names:
                img_orig = load_img(os.path.join(img_path, image_name))
                self.generate_augmentation(img_path, image_name, img_orig)
        print('augmentation finished')

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
            brightness_range=[0.3,0.9],
            horizontal_flip=True
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
