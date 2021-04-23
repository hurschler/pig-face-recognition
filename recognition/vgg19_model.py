import tensorflow as tf
import logging.config
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
import logging.config
import util.logger_init
from recognition.feature_extraction_model import FeatureExtractionModel


# Wrapper Class around Keras Model
class Vgg19(FeatureExtractionModel):

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("init Vgg19")
        self.height = 448
        self.width = 448
        self.model = self.define_model()

    def preprocessing_input(self, image):
        self.log.info('Start preprocessing Vgg19...')
        return tf.keras.applications.vgg19.preprocess_input(image)

    def define_model(self):
        inputs = layers.Input(shape=(self.width, self.height, 3))
        model = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            pooling='avg',
            classifier_activation="softmax",
        )
        # Freeze the pretrained weights
        model.trainable = False
        x = model.output
        model = tf.keras.Model(inputs, outputs=x, name="Vgg19")
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
        metrics = ['accuracy', 'mse', 'categorical_accuracy', 'top_k_categorical_accuracy']
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.summary()
        return model

    def getModel(self):
        return self.model

    def get_embeddings(self, img):
        return self.model(img)

    def get_target_size(self):
        return 448, 448

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def get_feature_vector_name(self):
        return 'data-vgg19.json'


    def remove_last_layer(self):
        # Remove Last Softmax layer and get model upto last flatten layer with outputs 2622 units
        self.model = Model(inputs=self.model.layers[0].input,
                           outputs=self.model.layers[-2].output)