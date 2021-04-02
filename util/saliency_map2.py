import tensorflow as tf
import tensorflow.keras as keras
import os
from matplotlib import pyplot as plt

import numpy as np
print('tensorflow {}'.format(tf.__version__))
print("keras {}".format(keras.__version__))

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# https://usmanr149.github.io/urmlblog/cnn/2020/05/01/Salincy-Maps.html
model = keras.applications.VGG16(weights='imagenet')
model.summary()

_img = keras.preprocessing.image.load_img('../sample/DSC_V1_6460_2238.JPG',target_size=(224,224))
plt.imshow(_img)
plt.show()

#preprocess image to get it into the right format for the model
img = keras.preprocessing.image.img_to_array(_img)
img = img.reshape((1, *img.shape))
y_pred = model.predict(img)
layers = [layer.output for layer in model.layers]
images = tf.Variable(img, dtype=float)

with tf.GradientTape() as tape:
    pred = model(images, training=False)
    class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
    loss = pred[0][class_idxs_sorted[0]]

grads = tape.gradient(loss, images)
grads.shape
dgrad_abs = tf.math.abs(grads)
dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
dgrad_max_.shape

## normalize to range between 0 and 1
arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

grad_eval.shape

fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].imshow(_img)
i = axes[1].imshow(grad_eval,cmap="jet",alpha=0.8)
fig.colorbar(i)
plt.show()

