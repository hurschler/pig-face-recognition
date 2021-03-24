import os

image_root_dir_path = r"D:\Users\avatar\OneDrive - Hochschule Luzern\bearbeitet_mit_label"

image_train_dir_path = os.path.join(image_root_dir_path, "train")

image_train_with_subdir_path = os.path.join(image_root_dir_path, "cropped-subdir-08-03-2021")

image_example_name = r"DSC_V1_6460_2238.JPG"

max_image_number = 10

keras_max_augmentation = 3