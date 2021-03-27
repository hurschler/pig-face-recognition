import logging.config
import util.logger_init

class MlModel:

    def getModel(self):
        return self.model

    def summary_print(self):
        self.model.summary()