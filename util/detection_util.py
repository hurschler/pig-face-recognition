import logging
import logging.config
import util.logger_init

log = logging.getLogger(__name__)


class DetectionUtil(object):

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("init DetectionUtil")

    def getPigName(self, image_name):
        start = "DSC_" + self.getSetVersion(image_name) + "_"
        end = "_"
        pig_name = (image_name.split(start)[1]).split(end)[0]
        return pig_name


    def getSetVersion(self, image_name):
        start = "DSC_"
        end = "_"
        s1 = image_name.split(start)[1]
        set_version = s1.split(end)[0]
        return set_version

