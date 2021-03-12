# example of horizontal shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from matplotlib import pyplot as plt

# based on https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/

# load the image
img = load_img('../sample/DSC_V1_6460_2238.JPG')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator

# good
# datagen = ImageDataGenerator(width_shift_range=[-500, 1000])
# datagen = ImageDataGenerator(brightness_range=[0.1,0.9]),
# datagen = ImageDataGenerator(rotation_range=45)
# datagen=ImageDataGenerator(zoom_range=[0.2,1.6])
# datagen=ImageDataGenerator(horizontal_flip=True)

datagen = ImageDataGenerator(
    width_shift_range=[-500, 1000],
    brightness_range=[0.1,0.9],
    rotation_range=45
)

# prepare iterator
it = datagen.flow(samples, batch_size=1)


# generate samples and plot
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # generate batch of images
    batch = it.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
    # plot raw pixel data
    pyplot.imshow(image)
# show the figure
pyplot.show()