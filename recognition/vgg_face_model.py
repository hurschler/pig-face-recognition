from keras import backend as K
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization
import os
import numpy as np
from matplotlib import pyplot
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import logging.config
from keras.layers import Input
from keras_vggface.vggface import VGGFace
import util.logger_init
import keras



# Wrapper Class around Keras Model
class VggFaceModel:

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("init VggFaceModel")
        self.sequential_model = self.define_model_vggface16_backend()
        # self.sequential_model = self.define_model_resnet_backend()
        # self.sequential_model = self.define_model_senet_backend()

        self.model = self.sequential_model


    def define_model_resnet_backend(self):
        return VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


    def define_model_senet_backend(self):
        return VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


    def define_model_vggface16_backend(self):
        # Define VGG_FACE_MODEL architecture
        # https://medium.com/analytics-vidhya/face-recognition-with-vgg-face-in-keras-96e6bc1951d5
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Convolution2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))

        return model

    def remove_last_layer(self):
        # Remove Last Softmax layer and get model upto last flatten layer with outputs 2622 units
        self.model = Model(inputs=self.sequential_model.layers[0].input,
                           outputs=self.sequential_model.layers[-2].output)

    def load_weights(self):
        self.sequential_model.load_weights('../model/vgg_face_weights.h5')


    def addAugmentationLayer(self):
        data_augmentation = keras.Sequential([
            keras.layers.experimental.preprocessing.RandomRotation(factor=0.4, fill_mode="wrap"),
            keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode="wrap"),
            keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            keras.layers.experimental.preprocessing.RandomContrast(factor=0.2),
            keras.layers.experimental.preprocessing.RandomHeight(factor=0.2),
            keras.layers.experimental.preprocessing.RandomWidth(factor=0.2)
        ])


    def vgg_face(self, img):
        return self.model(img)

    def debug_model(self, img_name):

        # crop_img = load_img(os.getcwd() + '/' + img_name, target_size=(224, 224))
        crop_img = load_img(img_name, target_size=(224, 224))
        crop_img = img_to_array(crop_img)
        crop_img = np.expand_dims(crop_img, axis=0)
        crop_img = preprocess_input(crop_img)

        # --------------   Debugging CNN ------------------------------
        # https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
        img_debug = load_img(img_name, target_size=(224, 224))
        # convert the image to an array
        img_debug = img_to_array(img_debug)
        # expand dimensions so that it represents a single 'sample'
        img_debug = np.expand_dims(crop_img, axis=0)
        # prepare the image (e.g. scale pixel values for the vgg)
        img_debug = preprocess_input(crop_img)
        # get feature map for first hidden layer
        # redefine model to output right after the first hidden layer
        model_with_feature_map = Model(inputs=self.sequential_model.inputs,
                                       outputs=self.sequential_model.layers[1].output)
        # get feature map for first hidden layer
        feature_maps = model_with_feature_map.predict(crop_img)
        # plot all 64 maps in an 8x8 squares
        square = 8
        ix = 1
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = pyplot.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                pyplot.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
                ix += 1
        # show the figure
        pyplot.savefig('../output/debug.jpg')
        pyplot.show()
        # --------------   Debugging CNN ------------------------------

    def get_embeddings(self, crop_img_name):
        # Get Embeddings
        # crop_img = load_img(os.getcwd() + '/' + crop_img_name, target_size=(224, 224))
        crop_img = load_img(crop_img_name, target_size=(224, 224))
        crop_img = img_to_array(crop_img)
        crop_img = np.expand_dims(crop_img, axis=0)
        crop_img = preprocess_input(crop_img)
        return self.model(crop_img)

