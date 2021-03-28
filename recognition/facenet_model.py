from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import util.detection_config as detection_config
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.platform import gfile
import logging.config
import util.logger_init


# Wrapper Class around Tensorflow Model
class FaceNetModel:

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("init FaceNetModel")
        # self.load_model()


    def get_embeddings_train(self, ml_data):
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=False)) as sess:
            self.load_model()
            init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
            sess.run(init_op)
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embedding_layer = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            coord = tf.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(coord=coord, sess=sess)

            img_path_crop = detection_config.output_path_cropped_rectangle
            pig_img_folders = os.listdir(img_path_crop)
            for i, pig_name in enumerate(pig_img_folders):
                ml_data.pig_dict[i] = pig_name
                image_names = os.listdir(os.path.join(img_path_crop, pig_name))
                for image_name in image_names:

                    # Get Embeddings
                    crop_img = load_img(os.path.join(img_path_crop, pig_name, image_name), target_size=(160, 160))
                    crop_img = img_to_array(crop_img)
                    crop_img = np.expand_dims(crop_img, axis=0)
                    crop_img = preprocess_input(crop_img)
                    img_tensor = tf.convert_to_tensor(crop_img)
                    label_tensor = tf.convert_to_tensor(pig_name)
                    batch_images, batch_labels = sess.run([img_tensor, label_tensor])
                    img_encode = sess.run(embedding_layer, feed_dict={images_placeholder: batch_images, phase_train_placeholder: False})
                    feature_vector = np.squeeze(K.eval(img_encode)).tolist()
                    ml_data.x_train.append(feature_vector)
                    ml_data.y_train.append(i)
                    print ('TRAIN pig-number: ', i, ' pig_name: ', pig_name, 'image_name:  ', image_name, 'length of Feature-Vector: ', len(feature_vector), ' Feature-Vector: ', feature_vector)


    def get_embeddings_test(self, ml_data):
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=False)) as sess:
            self.load_model()
            init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
            sess.run(init_op)
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embedding_layer = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            coord = tf.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(coord=coord, sess=sess)

            img_path_crop = detection_config.output_path_cropped_rectangle_test
            pig_img_folders = os.listdir(img_path_crop)
            for i, pig_name in enumerate(pig_img_folders):
                ml_data.pig_dict[i] = pig_name
                image_names = os.listdir(os.path.join(img_path_crop, pig_name))
                for image_name in image_names:

                    # Get Embeddings
                    crop_img = load_img(os.path.join(img_path_crop, pig_name, image_name), target_size=(160, 160))
                    crop_img = img_to_array(crop_img)
                    crop_img = np.expand_dims(crop_img, axis=0)
                    crop_img = preprocess_input(crop_img)
                    img_tensor = tf.convert_to_tensor(crop_img)
                    label_tensor = tf.convert_to_tensor(pig_name)
                    batch_images, batch_labels = sess.run([img_tensor, label_tensor])
                    img_encode = sess.run(embedding_layer, feed_dict={images_placeholder: batch_images, phase_train_placeholder: False})
                    feature_vector = np.squeeze(K.eval(img_encode)).tolist()
                    ml_data.x_test.append(feature_vector)
                    ml_data.y_test.append(i)
                    print ('TEST pig-number: ', i, ' pig_name: ', pig_name, 'image_name:  ', image_name, 'length of Feature-Vector: ', len(feature_vector), ' Feature-Vector: ', feature_vector)


    def get_embedding(self, crop_img_name):
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=False)) as sess:
            self.load_model()
            init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
            sess.run(init_op)
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embedding_layer = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            coord = tf.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(coord=coord, sess=sess)

            # Get Embeddings
            crop_img = load_img(crop_img_name, target_size=(160, 160))
            crop_img = img_to_array(crop_img)
            crop_img = np.expand_dims(crop_img, axis=0)
            crop_img = preprocess_input(crop_img)
            img_tensor = tf.convert_to_tensor(crop_img)
            pig_name = 'PIG-NAME'
            label_tensor = tf.convert_to_tensor(pig_name)
            batch_images, batch_labels = sess.run([img_tensor, label_tensor])
            img_encode = sess.run(embedding_layer, feed_dict={images_placeholder: batch_images, phase_train_placeholder: False})
            # feature_vector = np.squeeze(K.eval(img_encode)).tolist()
            return img_encode


    def load_model(self):
        model_filepath = "D:\\Users\\avatar\\PycharmProjects\\Python-Deep-Learning-Projects\\Chapter10\\pre-model\\20180402-114759.pb"
        model_exp = os.path.expanduser(model_filepath)
        if os.path.isfile(model_exp):
            logging.info('Model filename: %s' % model_exp)
            with gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        else:
            self.log.error('Missing model file. Exiting: ' + model_filepath)



