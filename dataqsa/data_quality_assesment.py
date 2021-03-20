import util.preprocessing as preprocessing
import util.detection_util as detection_util
import metadataextractor
import util.config as config
import logging.config
import imutils
import pandas as pd
import numpy as np
import cv2
import sys
import os
import util.logger_init
from matplotlib import pyplot as plt
from PIL import ImageChops
from PIL import Image as Pil_Image
import sys



log = logging.getLogger(__name__)

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)



log.info("read images from directory")
pre = preprocessing.Preprocessing()
detect = detection_util.DetectionUtil()
img_dic = pre.readImages()

df = pd.DataFrame(columns=['id', 'imageName', 'type', 'fsize', 'pigname', 'setversion', 'createdate', 'img_width',
                           'img_height', 'sharpness', 'flash', 'bright', 'contrast', 'sex', 'weight', 'age', 'y_max_hist'])
i = 0


def showOpenCvImage(img_before, img_after, image_name):
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
    plt.title(image_name)
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
    plt.title(image_name)
    plt.show()


def findArc(img, th):
    img = imutils.resize(img, height=224)
    res = img.copy()
    ## convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.bilateralFilter(gray, 10, 50, 0)
    ## threshold the gray
    th, threshed = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
    ## Find contours on the binary threshed image
    edged = cv2.Canny(gray, 30, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    ## calcualte
    for cnt in cnts:
        arclen = cv2.arcLength(cnt, True)

    area = cv2.contourArea(cnt)
    cv2.drawContours(res, [cnt], -1, (0, 255, 0), 3, cv2.LINE_AA)
    print("Length: {:.3f}nArea: {:.3f}".format(arclen, area))
    return res


def compareImage(img_old, img_new):
    diff = ImageChops.difference(img_old, img_new)
    return diff


def findContourWithColor(img):
    # lower_yellow = (50, 50, 50)
    # upper_yellow = (140, 160, 245)
    lower_color = (80, 80, 80)
    upper_color = (140, 160, 245)
    # konvertiere Frame in HSV-Farbraum, um besser nach Farb-Ranges filtern zu können
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(frame, lower_color, upper_color)
    cnts = cv2.findContours(mask, cv2.cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    area = 0
    for cnt in cnts:
        arclen = cv2.arcLength(cnt, True)
        area = area + cv2.contourArea(cnt)
        cv2.drawContours(frame, cnt, -1, (0, 255, 0), 3, cv2.LINE_AA)

    print("dirty-area: ", area)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
    return frame


old_img = None

for key in img_dic:
    img = img_dic[key]
    img_with_path = os.path.join(config.image_train_dir_path, key)
    size = os.stat(img_with_path).st_size
    pig_name = detect.getPigName(key)
    set_version = detect.getSetVersion(key)
    meta = metadataextractor.MetadataExtractor(img_with_path)
    create_date = meta.getCreateDate()
    img_width = meta.getImageWidth()
    img_height = meta.getImageHeight()
    flash = meta.getFlashMode()
    shrp = pre.computeSharpness(img)
    bright = pre.getBrightness(img)
    contrast = pre.getContrast(img)
    sex = 1
    age = 90
    weight = 100

    img_after = pre.automatic_brightness_and_contrast(img)

    # res = findArc(img, 10)
    # res = findContourWithColor(img)


    plt.rcParams["figure.figsize"] = (8, 6)
    plt.subplot(121)
    y, x, _ = plt.hist(img.ravel(), bins = 256, color = 'orange', )
    plt.hist(img[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
    plt.hist(img[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
    plt.hist(img[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
    plt.xlabel('Intensity Value')
    plt.ylabel('Count')
    plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.figure(1)
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(key + ' Y-Max: ' + str(y.max()))
    # plt.show()
    y_max_hist = y.max()

    # showOpenCvImage(img, np.asarray(img_after), key)

    df.loc[str(i), :] = [i, key, 'jpg', size, pig_name, set_version, create_date, img_width, img_height, shrp, flash,
                         bright, contrast, sex, weight, age, y_max_hist]
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    i = i + 1

pd.set_option('display.max_columns', 7)
print(df.head().to_string())

df.describe()
df['sharpness'].hist(bins=100)
plt.xlabel("id", fontsize=15)
plt.ylabel("sharpness", fontsize=15)
plt.xscale('log')
# plt.show()
