import os
import cv2
import util.config as config
import glob
import logging
import logging.config
import util.logger_init

from matplotlib import pyplot as plt

log = logging.getLogger(__name__)


class Preprocessing(object):

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("init Preprocessing")

    def readImage(self, image_name=config.image_example_name):
        log.info("readImage")
        dir_path = config.image_train_dir_path
        full_path = os.path.join(dir_path, image_name)
        image = cv2.imread(full_path)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        plt.imshow(image_gray, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        cv2.waitKey(0)

    def readImages(self):
        log.info("readImages")
        dir_path = config.image_train_dir_path
        log.info("image_dir: " + dir_path)
        img_dic = {}
        i = 0
        files = glob.glob(dir_path + r"\*.JPG")
        for imageFullFileName in files:
            if i >= config.max_image_number:
                break
            log.debug(imageFullFileName)
            image = cv2.imread(imageFullFileName)
            image_file_name = os.path.basename(imageFullFileName)
            log.info(image_file_name)
            img_dic[image_file_name] = image
            i = i + 1

        return img_dic

    def readImagesWithSubDir(self):
        log.info("readImages with sub dir")
        dir_path = config.image_train_with_subdir_path
        log.info("image_dir: " + dir_path)
        img_dic = {}
        files = glob.glob(dir_path + r"/*/")
        for path_element in files:
            if os.path.isdir(path_element):
                log.debug(path_element)
                files_on_sub_dir = glob.glob(os.path.join(dir_path, path_element) + r"\*.JPG")
                for image_on_sub_dir in files_on_sub_dir:
                    log.info("image on sub dir: " + image_on_sub_dir)
                    image = cv2.imread(image_on_sub_dir)
                    img_dic[image_on_sub_dir] = image

        return img_dic


    def toGrey(self, img_dic):
        log.info("preprocessing toGrey")
        img_dic_new = {}
        for key in img_dic:
            image = img_dic[key]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            img_dic_new[key] = image_gray

        return img_dic_new

    def scale(self, img_dic):
        log.info("preprocessing scale Images")
        img_dic_new = {}
        dim = (224, 224)
        for key in img_dic:
            image = img_dic[key]
            img_dic_new[key] = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        return img_dic_new


    def toEqualizeHist(self, img_dic):
        log.info("preprocessing equalize histogram")
        for key in img_dic:
            image = img_dic[key]
            image = cv2.equalizeHist(image)
            img_dic[key] = image

        return img_dic

    # Sharpness / Blur detection
    def computeSharpness(self, img_dic):
        log.info("preprocessing compute sharpness")
        for key in img_dic:
            image = img_dic[key]
            s = cv2.Laplacian(image, cv2.CV_64F).var()
            log.info ("Sharpness: " + key + " " + str(s))

    # Sharpness / Blur detection
    def computeSharpness(self, cv2_image):
        sharpness = cv2.Laplacian(cv2_image, cv2.CV_64F).var()
        return sharpness;

    def show(self, img_dic, gray=True):
            log.info("show")
            for key in img_dic:
                image = img_dic[key]
                if gray:
                    plt.imshow(image, cmap='gray', interpolation='bicubic')
                else:
                    plt.imshow(image)
                plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
                plt.show()
                cv2.waitKey(0)
