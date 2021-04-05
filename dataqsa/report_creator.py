import glob
import logging.config
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import util.preprocessing as preprocessing
import xlrd


import cv2

import util.logger_init
import util.config as config
from dataqsa import metadataextractor
from util import detection_util
from util.detection_util import DetectionUtil
import openpyxl

class ReportCreator():

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("init ReportCreator")
        self.excel_file_path_full_name = '../sample/train.xlsx'
        self.df = None
        # self.readExcelToPanda()
        # self.build_image_dictionary()

    def build_image_dictionary(self):
        self.log.info("read Information about Images")
        dir_path = config.image_train_dir_path
        self.log.info("image_dir to build dictionary: " + dir_path)
        i = 0
        files = [x for x in os.listdir(dir_path) if (x.endswith('.jpg') or x.endswith('.JPG') or x.endswith('.png') or x.endswith('.PNG'))]
        detection_util = DetectionUtil()
        for imageFullFileName in files:
            imageFullFileName = os.path.join(dir_path, imageFullFileName)
            if i >= config.max_image_number:
                break
            image_file_name = os.path.basename(imageFullFileName)
            pig_name = detection_util.getPigName(image_file_name)
            self.addRow(imageFullFileName, image_name=image_file_name, pig_name=pig_name)
            self.log.info(imageFullFileName)
            i = i + 1



    def readExcelToPanda(self):
        self.df = pd.read_excel(self.excel_file_path_full_name)
        print (self.df.head())

    def writePandaToExcel(self):
        print (self.df.head())
        self.df.to_excel(self.excel_file_path_full_name, index=False)


    def getType(self, img_name_with_path):
        return 'jpg'

    def getSize(self, img_name_with_path):
        return os.stat(img_name_with_path).st_size

    def getPigName(self, img_name_with_path):
        image_file_name = os.path.basename(img_name_with_path)
        detect = detection_util.DetectionUtil()
        return detect.getPigName(image_file_name)

    def getVersion(self, img_name_with_path):
        image_file_name = os.path.basename(img_name_with_path)
        detect = detection_util.DetectionUtil()
        set_version = detect.getSetVersion(image_file_name)
        return set_version

    def getCreationDate(self, img_name_with_path):
        meta = metadataextractor.MetadataExtractor(img_name_with_path)
        return meta.getCreateDate()


    def getWidth(self, img_name_with_path):
        meta = metadataextractor.MetadataExtractor(img_name_with_path)
        return meta.getImageWidth()

    def getHeight(self, img_name_with_path):
        meta = metadataextractor.MetadataExtractor(img_name_with_path)
        return  meta.getImageHeight()

    def getFlashMode(self, img_name_with_path):
        meta = metadataextractor.MetadataExtractor(img_name_with_path)
        return  meta.getFlashMode()

    def getSharpness(self, img_name_with_path):
        image = cv2.imread(img_name_with_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pre = preprocessing.Preprocessing()
        sharpness = pre.computeSharpness(image_rgb)
        return  sharpness

    def getBrightness(self, img_name_with_path):
        image = cv2.imread(img_name_with_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pre = preprocessing.Preprocessing()
        bright = pre.getBrightness(image_rgb)
        return bright


    def getContrast(self, img_name_with_path):
        image = cv2.imread(img_name_with_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pre = preprocessing.Preprocessing()
        contrast = pre.getContrast(image_rgb)
        return contrast


    def addRow(self, img_name_with_path, image_name=None, pig_name=None):
        new_row = {
            'image_name':image_name,
            'type': self.getType(img_name_with_path),
            'pig_name': self.getPigName(img_name_with_path),
            'setversion': self.getVersion(img_name_with_path),
            'createdate': self.getCreationDate(img_name_with_path),
            'img_width': self.getWidth(img_name_with_path),
            'img_height': self.getHeight(img_name_with_path),
            'sharpness': self.getSharpness(img_name_with_path),
            'flash': self.getFlashMode(img_name_with_path),
            'bright': self.getBrightness(img_name_with_path),
            'contrast': self.getContrast(img_name_with_path),
            'sex': 'm',
            'weight': '0',
            'age': '0',
            'perspective': '1',
            'full_pig': '1'
        }
        self.df = self.df.append(new_row, ignore_index=True)


    def mergeRows(self):
        # Todo implement
        return


report_creator = ReportCreator()
report_creator.readExcelToPanda()
report_creator.build_image_dictionary()
report_creator.writePandaToExcel()

