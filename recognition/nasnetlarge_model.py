import tensorflow as tf
import logging.config
from tensorflow.keras import layers
import logging.config
import util.logger_init


# Wrapper Class around Keras Model
from recognition.feature_extraction_model import FeatureExtractionModel


class NasNetLarge(FeatureExtractionModel):

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("init NasNetLarge")
        self.model = self.define_model()

    def define_model(self):
        inputs = layers.Input(shape=(331, 331, 3))
        model = tf.keras.applications.NASNetLarge(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            # pooling='avg',
            pooling='avg',
        )
        # Freeze the pretrained weights
        model.trainable = False
        x = model.output
        model = tf.keras.Model(inputs, outputs=x, name="NasNetLarge")
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
        metrics = ['accuracy', 'mse', 'categorical_accuracy', 'top_k_categorical_accuracy']
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def getModel(self):
        return self.model

    def nasnet_model(self, img):
        return self.model(img)

    def preprocessing_input(self, image):
        """Returns the data format of the NasNetLarge """
        self.log.info('Starting preprocessing NasNetLarge net model...')
        return tf.keras.applications.nasnet.preprocess_input(image)

    def get_target_size(self):
        return 331, 331

    def get_feature_vector_name(self):
        return 'data-nasnetlarge.json'