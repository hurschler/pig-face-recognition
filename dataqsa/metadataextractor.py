import sys
import exifread
import logging.config
import util.logger_init


class MetadataExtractor:

    def __init__(self, file_name_with_path):
        self.log = logging.getLogger(__name__)
        # open('../sample/DSC_V1_6460_2238.JPG', 'rb')
        file = open(file_name_with_path, 'rb')
        self.tags = exifread.process_file(file)

    def showAllKey(self):
        for tmp in self.tags.keys():
            self.log.info(str(tmp) + " = " + str(self.tags[tmp]))

    def getCreateDate(self):
        meta_aufnahme_datum = self.tags['EXIF DateTimeOriginal'].values
        return meta_aufnahme_datum

    def getImageWidth(self):
        meta_aufnahme_width = self.tags['EXIF ExifImageWidth'].values
        m1 = str(meta_aufnahme_width).split('[')[1]
        m1 = m1.split(']')[0]
        return m1

    def getImageHeight(self):
        meta_aufnahme_height = self.tags['EXIF ExifImageLength'].values
        m1 = str(meta_aufnahme_height).split('[')[1]
        m1 = m1.split(']')[0]
        return m1

    def getIso(self):
        meta_aufnahme_iso = self.tags['EXIF ISOSpeedRatings'].values
        return meta_aufnahme_iso
