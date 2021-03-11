import logging
import logging.config
import util.logger_init

log = logging.getLogger(__name__)


class DetectionUtil(object):

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("init DetectionUtil")

    def getPigName(self, image_name):
        start = "DSC_V1_"
        end = "_"
        pig_name = (image_name.split(start)[1]).split(end)[0]
        return pig_name

