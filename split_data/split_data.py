r"""Split raw dataset to train and test set with ratio.

Example usage:
    python3 split_data.py \
        --data_dir=/home/user/dataset \
        --ratio=7
"""

import argparse
import os
import shutil
import random
import pathlib

parser = argparse.ArgumentParser(
    description='Split raw dataset to train and test set with ratio')
parser.add_argument('-d', '--data-dir', type=str, required=True,
                    help='path to folder contain the data')
parser.add_argument('-i', '--image-name', default='images', type=str,
                    help='the name of image folder')
parser.add_argument('-x', '--xml-name', default='xml', type=str,
                    help='the name of xml folder')
parser.add_argument('-r', '--ratio', default='7', type=int,
                    help='ratio for train/test, default is 70 percent train'
                    ' and 30 percent test')
parser.add_argument('-o', '--output-dir', required=True, type=str,
                    help='path to folder contain the output data')

args = parser.parse_args()

IMAGE_EXTENSIONS = ['.jpg']
XML_EXTENSIONS = ['.xml']


def scanFiles(folder_path, extensions):
    '''
    folder_path: path to directory contain files.
    extansions: extensions (array of extension) of files.
    Return:
        array of file path.
    '''
    scanned_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(tuple(extensions)):
                relativePath = os.path.join(root, file)
                path = os.path.abspath(relativePath)
                scanned_files.append(path)
    scanned_files.sort(key=lambda x: x.lower())
    # scanned_files = random.sample(scanned_files, len(scanned_files))
    return scanned_files


def copy_files(files, dest_path):
    for file in files:
        file_name = os.path.basename(file)
        shutil.copy(file, os.path.join(dest_path, file_name))


if __name__ == '__main__':
    radio = int(args.ratio)

    input_path = args.data_dir
    image_folder_name = args.image_name
    image_folder_path = os.path.join(input_path, image_folder_name)

    xml_folder_name = args.xml_name
    xml_folder_path = os.path.join(input_path, xml_folder_name)

    output_path = args.output_dir

    output_train_image_path = os.path.join(
        output_path, 'train', image_folder_name)
    output_train_xml_path = os.path.join(output_path, 'train', xml_folder_name)

    output_test_image_path = os.path.join(
        output_path, 'test', image_folder_name)
    output_test_xml_path = os.path.join(output_path, 'test', xml_folder_name)

    # create folder if it's not exist.
    pathlib.Path(output_train_image_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_train_xml_path).mkdir(parents=True, exist_ok=True)

    pathlib.Path(output_test_xml_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_test_image_path).mkdir(parents=True, exist_ok=True)

    images = scanFiles(image_folder_path, IMAGE_EXTENSIONS)
    xmls = scanFiles(xml_folder_path, XML_EXTENSIONS)

    print(images[-1])
    print(xmls[-1])

    # randomizing two lists and maintaining order
    combined = list(zip(images, xmls))
    random.shuffle(combined)
    images[:], xmls[:] = zip(*combined)

    print(images[-1])
    print(xmls[-1])

    print('images: {} <-> xmls: {}'.format(len(images), len(xmls)))
    assert len(images) == len(xmls)

    assert radio <= 10

    train_num = int(len(images) * (radio/10))
    print(train_num)
    print('total: {}, train num: {}, test num: {}'.format(
        len(images), train_num, len(images) - train_num))

    train_images = images[:train_num]
    train_xmls = xmls[:train_num]

    test_images = images[train_num:]
    test_xmls = xmls[train_num:]

    # copy files
    copy_files(train_images, output_train_image_path)
    copy_files(train_xmls, output_train_xml_path)

    copy_files(test_images, output_test_image_path)
    copy_files(test_xmls, output_test_xml_path)
