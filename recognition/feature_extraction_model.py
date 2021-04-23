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
        
