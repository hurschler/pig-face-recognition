import logging.config
import util.logger_init


class FeatureExtractionModel:

    def __init__(self, ml_data):
        self.log = logging.getLogger(__name__)
        self.log.info("Init FeatureExtractionModel: " + __name__)

    def getInputDimension(self):
        return ''

    def getEmbedding(self):
        return ''

    def preprocessing_input(self, img):
        self.log.info('Start preprocessing FeatureExtractionModel...')

    def get_target_size(self):
        return 224, 224

    def get_embeddings(self):
        self.log.info('Get embeddings...')

    def load_weights(self):
        # self.sequential_model.load_weights('../model/vgg_face_weights.h5')$
        # Todo get_target_size load_weights
        print('Todo get_target_size load_weights')

    def save_weights(self):
        """Saves the weights in the model folder"""
        self.log.info('Saving weights...')
        self.model.save_weights('../model/inception_resnet_v2.h5')

    def get_embeddings(self, img):
        """Returns the embeddings from loaded image"""
        return self.model(img)
