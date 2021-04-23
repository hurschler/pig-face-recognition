import tensorflow as tf
import logging.config
import datetime
from tensorflow.python.keras.callbacks_v1 import TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization
from tensorflow.keras.models import Sequential, Model
import logging.config
import util.logger_init


class ResNetModel:

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("init ResNetModel")
        self.model = self.define_model()


    def define_model(self):
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=(224, 224, 3)
        )

        for layer in [l for l in base_model.layers if 'conv5' not in l.name]:
            layer.trainable = False

        for layer in [l for l in base_model.layers if 'conv5' in l.name or 'conv4' in l.name or l.name == 'probs']:
            layer.trainable = True

        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(20, activation="softmax"))
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        metrics = ['accuracy', 'mse', 'categorical_accuracy', 'top_k_categorical_accuracy']
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)
        self.log.info(model.summary())
        return model


    def fit(self, train_generator, validation_generator):
        logdir = "../logs/recognition/logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH",
                                                         circular_buffer_size=-1)
        self.checkpoint_path = '../model/face_model'
        lr_scheduler = LearningRateScheduler(self.scheduler)

        callb = [
            lr_scheduler,
            TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=False, write_images=True, update_freq='epoch',
                          profile_batch=2, embeddings_freq=0, embeddings_metadata=None),
            ModelCheckpoint(self.checkpoint_path, save_weights_only=True, save_best_only=True, monitor="val_loss",
                            verbose=1),
        ]
        self.model.summary()
        self.model.fit_generator(train_generator, epochs=10, callbacks= callb,  validation_data=validation_generator)

    def get_model(self):
        return self.model

    def load_weights(self):
        self.log.info('Loading weights...')
        self.model.load_weights('../model/resnet50_v2.h5')

    def save_weights(self):
        self.log.info('Saving weights...')
        self.model.save_weights('../model/resnet50_v2.h5')

    def getResnet_face(self, img):
        return self.model(img)

    def scheduler(self, epoch):
        return 0.001 * 0.95 ** epoch

