import tensorflow as tf
import logging.config
from tensorflow.keras import layers
import logging.config
import util.logger_init
from recognition.feature_extraction_model import FeatureExtractionModel


class ResNetModel(FeatureExtractionModel):

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("init ResNetModel")
        self.model = self.define_model()

    def define_model(self):
        h, w = self.get_target_size()
        inputs = layers.Input(shape=(h, w, 3))
        model = tf.keras.applications.ResNet152V2(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
        )

        x = model.output
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        model = tf.keras.Model(inputs, outputs=x, name="ResNet152V2")
        # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        metrics = ['accuracy', 'mse', 'categorical_accuracy', 'top_k_categorical_accuracy']
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def getResnet_face(self, img):
        return self.model(img)

    def preprocessing_input(self, image):
        """Returns the data format of the ResNetModel """
        self.log.info('Starting preprocessing ResNetModel net model...')
        return tf.keras.applications.resnet_v2.preprocess_input(image)

    def get_target_size(self):
        return 224, 224

    def get_feature_vector_name(self):
        return 'data-resnet152v2.json'
