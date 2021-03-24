import os
import configparser

local_property_file = '../local.properties'
build_property_file = '../build.properties'

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

image_root_dir_path = r"D:\Users\avatar\OneDrive - Hochschule Luzern\bearbeitet_mit_label"

image_train_dir_path = os.path.join(image_root_dir_path, "train")

image_train_with_subdir_path = os.path.join(image_root_dir_path, "cropped-subdir-08-03-2021")

image_example_name = r"DSC_V1_6460_2238.JPG"

max_image_number = 10

keras_max_augmentation = 3


#merge all into one config dictionary
config = read_config([local_property_file, build_property_file])

keras_max_augmentation = int(config.get('glob', 'keras_max_augmentation'))
image_root_dir_path = config.get('glob', 'image_root_dir_path')
image_sample_full_path = config.get('glob', 'image_sample_full_path')
build_server_path = config.get('glob', 'build_server_path')