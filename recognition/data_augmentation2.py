from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import os
import datetime

from tensorflow.python.keras.callbacks import ModelCheckpoint

# Example from here
# https://stepup.ai/train_data_augmentation_keras/
# https://github.com/dufourpascal/stepupai/blob/master/tutorials/data_augmentation/train_data_augmentation_keras.ipynb

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

#Loading the image and coverting into Byte
img_array= load_img('../sample/DSC_V1_6460_2238.JPG')


def visualize_data(images, categories, class_names):
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor('white')
    for i in range(3 * 7):
        plt.subplot(3, 7, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i])
        class_index = categories[i].argmax()
        plt.xlabel(class_names[class_index])
    plt.show()


def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(class_names)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
y_train = to_categorical(y_train, num_classes)
x_test = x_test / 255.0
y_test = to_categorical(y_test, num_classes)
# visualize_data(x_train, y_train, class_names)

batch_size = 32
epochs = 16
# m_no_aug = create_model()
# m_no_aug.summary()

logdir = "../logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
checkpoint_path = os.path.join('../model', 'test_checkpoint.h5')

callb = [
    ModelCheckpoint(checkpoint_path,save_weights_only=True,save_best_only = True, monitor = "val_loss", verbose = 1),
    tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=True, update_freq='epoch', profile_batch=2, embeddings_freq=0, embeddings_metadata=None),
]

# m_no_aug = create_model()
# m_no_aug.summary()
# history_no_aug = m_no_aug.fit(x_train, y_train, epochs=epochs, callbacks=callb, batch_size=batch_size,validation_data=(x_test, y_test))
# loss_no_aug, acc_no_aug = m_no_aug.evaluate(x_test,  y_test)

width_shift = 3/32
height_shift = 3/32
flip = True

datagen = ImageDataGenerator(
    horizontal_flip=flip,
    width_shift_range=width_shift,
    height_shift_range=height_shift,
)
datagen.fit(x_train)

it = datagen.flow(x_train, y_train, shuffle=False)
batch_images, batch_labels = next(it)
# visualize_data(batch_images, batch_labels, class_names)

m_aug = create_model()
datagen.fit(x_train)

history_aug = m_aug.fit(datagen.flow(x_train, y_train, batch_size=batch_size),epochs=epochs, callbacks=callb, validation_data=(x_test, y_test))
loss_aug, acc_aug = m_aug.evaluate(x_test,  y_test)


