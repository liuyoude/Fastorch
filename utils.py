"""
functional functions
"""
import os
import glob
import yaml
import csv
import logging


def load_yaml(file_path='./config.yaml'):
    with open(file_path) as f:
        params = yaml.safe_load(f)
    return params


def save_yaml_file(file_path, data: dict):
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f, encoding='utf-8', allow_unicode=True)


def save_csv(file_path, data: list):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)


def save_model_state_dict(file_path, epoch=None, net=None, optimizer=None):
    import torch
    state_dict = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict() if optimizer else None,
        'model': net.state_dict() if net else None,
    }
    torch.save(state_dict, file_path)


def get_logger(filename):
    logging.basicConfig(filename=filename, level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler)
    return logger

def get_filename_list(dir_path, ext=None):
    """
    find all extention files under directory
    :param dir_path: directory path
    :param ext: extention name, like wav, png...
    :return: files path list
    """
    filename_list = []
    ext = ext if ext else '*'
    for root, dirs, files in os.walk(dir_path):
        file_path_pattern = os.path.join(root, f'*.{ext}')
        files = sorted(glob.glob(file_path_pattern))
        filename_list += files
    return filename_list


if __name__ == '__main__':
    print(get_filename_list('../Fastorch', ext='py'))
