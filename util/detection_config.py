import os
import configparser

local_property_file = '../local.properties'
build_property_file = '/home/runner/work/pig-face-recognition/pig-face-recognition/build.properties'

# a simple function to read an array of configuration files into a config object
def read_config(cfg_files):
    if(cfg_files != None):
        config = configparser.RawConfigParser()

        # merges all files into a single config
        for i, cfg_file in enumerate(cfg_files):
            if(os.path.exists(cfg_file)):
                if(cfg_file == local_property_file):
                    config.read(cfg_file)
                    break
                if(os.path.exists(cfg_file)):
                    config.read(cfg_file)

        return config

#merge all into one config dictionary
config = read_config([local_property_file, build_property_file])

# Detection Config

image_root_dir_path = r"D:\Users\avatar\OneDrive - Hochschule Luzern\bearbeitet_mit_label"

image_train_dir_path = os.path.join(image_root_dir_path, "train")

output_path_cropped = r"D:\Users\avatar\OneDrive - Hochschule Luzern\bearbeitet_mit_label\cropped-subdir-08-03-2021"

# output_path_cropped_rectangle = r"D:\Users\avatar\OneDrive - Hochschule Luzern\bearbeitet_mit_label\cropped-rectangle-subdir-08-03-2021"

# output_path_cropped_rectangle = r"G:\temp\pig-face-22-03-2021"
# output_path_cropped_rectangle_test = r"G:\temp\pig-face-22-03-2021-test"

output_path_cropped_rectangle = r"G:\temp\pig-face-rectangle"
output_path_cropped_rectangle_test = r"G:\temp\pig-face-rectangle-test"


image_example_name = r"DSC_V1_6460_2238.jpg"

max_image_number = 30

# image_upload_dir_path = r"D:\Users\avatar\PycharmProjects\pig-face-recognition\app\upload"
image_upload_dir_path = config.get('glob', 'image_upload_dir_path')

# output_dir_path = r"D:\Users\avatar\PycharmProjects\pig-face-recognition\output"
output_dir_path = config.get('glob', 'output_dir_path')