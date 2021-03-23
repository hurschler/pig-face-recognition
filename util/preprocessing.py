import os
import cv2
import util.config as config
import glob
import numpy as np
import logging
import logging.config
from PIL import Image as Pil_Image
from PIL import ImageStat
import util.logger_init

from matplotlib import pyplot as plt

log = logging.getLogger(__name__)


class Preprocessing(object):

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.info("init Preprocessing")

    def readImage(self, img_path, img_name):
        img_name_full = os.path.join(img_path, img_name)
        log.info("readImage: " + img_name_full)
        image = cv2.imread(img_name_full)
        return image

    def replaceColor(self, img_opencv, r, g, b):
        lower_black = np.array([r,g,b], dtype = "uint16")
        upper_black = np.array([r,g,b], dtype = "uint16")
        black_mask = cv2.inRange(img_opencv, lower_black, upper_black)
        img_opencv[np.where((img_opencv == [b,g,r]).all(axis = 2))] = [0,0,0]
        return img_opencv

    def replaceColorBlueWithBlack(self, img_opencv):
        img = self.replaceColor(img_opencv, 0, 0, 194)
        img = self.replaceColor(img_opencv, 5, 6, 150)
        return img

    def readImageToGray(self, image_name=config.image_example_name):
            log.info("readImage: " + image_name)
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
            log.debug(image_file_name)
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


    # Automatic brightness and contrast optimization with optional histogram clipping
    def automatic_brightness_and_contrast(self, image, clip_hist_percent=1):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Calculate grayscale histogram
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_size = len(hist)
        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0
        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha
        auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return auto_result


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
    # Zweite Ableitung, stärkste Krümmung
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


    def getBrightness(self, cv2_image):
        img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        im_pil = Pil_Image.fromarray(img)
        # Only get the Luminance value (-> to Gray)
        im_pil = im_pil.convert('L')
        stat = ImageStat.Stat(im_pil)
        return stat.mean[0]

    def getContrast(self, cv2_image):
        Y = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2YUV)[:,:,0]
        # compute min and max of Y
        min = np.min(Y)
        max = np.max(Y)
        # compute contrast
        contrast = (max-min)/(max+min)
        return contrast


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
