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


config = read_config([local_property_file, build_property_file])

image_root_dir_path = config.get('glob', 'image_root_dir_path')

image_train_dir_path = os.path.join(image_root_dir_path, "train")

image_train_with_subdir_path = os.path.join(image_root_dir_path, "cropped-subdir-08-03-2021")

image_test_with_subdir_path = os.path.join(r'G:\temp\pig-face-rectangle-test')

image_example_name = r"DSC_V1_6460_2238.JPG"

image_example_blue_name = r"0-DSC_V1_6494_2109.JPG-blue-background.jpg"

build_server_path = r'/home/runner/work/pig-face-recognition/pig-face-recognition'

output_path = r'/Users/patrickrichner/Desktop/FH/11.Semester/Bda2021/pig-face-recognition/output'

image_sample_path = r'/Users/patrickrichner/Desktop/FH/11.Semester/Bda2021/pig-face-recognition/sample'

max_image_number = 2000

# Recognition
output_path_cropped_rectangle_test = config.get('glob', 'output_path_cropped_rectangle_test')
output_path_cropped_rectangle = config.get('glob', 'output_path_cropped_rectangle')



# Detection
output_dir_path = config.get('glob', 'output_dir_path')
image_upload_dir_path = config.get('glob', 'image_upload_dir_path')

# Test
test_image_folder = config.get('glob', 'test_image_folder')