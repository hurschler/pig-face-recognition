import pandas as pd
import numpy as np
import logging.config

import util.config
from augmentation import aug_util
from util import logger_init

log = logging.getLogger(__name__)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
IMAGE_PATH = util.config.output_path_cropped_rectangle_test
IMAGE_STATISTIC_PATH = r'../sample/train.xlsx'


def read_image_statistic():
    """Reads the statistic from a excel file"""
    df = pd.read_excel(IMAGE_STATISTIC_PATH)
    log.info(df.head().to_string())
    df.head(5)

    df = df.astype({'image_name': 'str', 'perspective': 'str', 'full_pig_face': 'int'})
    df['perspective'] = pd.Categorical(df.perspective)
    df.info()

    index = df.index
    log.info('Image Count all:' + str(len(index)))
    return df


def filter_dirty_images(df):
    """
    Filters all images which are in certain (bad) range
    """
    df_q = df.query('('
                    '(perspective == "sr") or '
                    '(perspective == "sl")) or '
                    '(over_exposed == 1) or '
                    '(missing_element.notnull('
                    ')) or '
                    '(bright < 30) or '
                    '(sharpness < 11)'
                    )
    return df_q


def remove_dirty_images(image_name, path=IMAGE_PATH):
    """
    Removes all evaluated dirty images
    Necessary calls:
     - df = read_image_statistic()
     - df_q = filter_dirty_images(df)
     - [remove_dirty_images(image_name) for image_name in df_q['image_name']]
    """
    pattern = '*' + image_name + '*'
    aug_util.clean_augmented_images(path=path, pattern=pattern)
    log.info('Start searching process for image nameof images with the pattern ' + pattern + '...')


df = read_image_statistic()
df_q = filter_dirty_images(df)
[remove_dirty_images(image_name) for image_name in df_q['image_name']]
