import util.preprocessing as preprocessing
import util.detection_util as detection_util
import metadataextractor
import util.config as config
import logging.config
import pandas as pd
import numpy as np
import cv2
import sys
import os
import util.logger_init



log = logging.getLogger(__name__)

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

log.info("read images from directory")
pre = preprocessing.Preprocessing()
detect = detection_util.DetectionUtil()
img_dic = pre.readImages()

df = pd.DataFrame(columns=['id', 'imageName', 'type', 'fsize', 'pigname', 'setversion', 'createdate', 'img_width', 'img_height', 'sharpness'])
i = 0

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
    shrp = pre.computeSharpness(img)
    meta.showAllKey()

    df.loc[str(i),:] = [i, key, 'jpg', size, pig_name, set_version, create_date, img_width, img_height, shrp]
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    i = i + 1


pd.set_option('display.max_columns', 7)
print(df.head().to_string())