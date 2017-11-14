import os
import sys
import logging
import numpy as np
import scipy.io as sio
from scipy.misc import imread
from scipy.misc import imshow


ZIP_FILE_URL = 'http://tamaraberg.com/faceDataset/faceData.tar.gz'
ZIP_FILE_PATH = 'faceData.tar.gz'
UNZIPPED_FILE_PATH = './'
MAT_FILE = 'facedata/FacesInTheWild.mat'

def download_and_unzip():
    """
    Downlaods the zip file and unzips the dataset.
    :return:
    """
    logger = logging.getLogger()
    if os.path.exists(MAT_FILE):
        # unzipped path exists
        logger.info('Unzipped dataset exists')
        return

    if not os.path.exists(ZIP_FILE_PATH):
        # donwload zip file
        logger.info('Downloading faces-in-the-wild dataset zip file')
        curl_cmd = 'curl {} -O {}'.format(ZIP_FILE_URL, ZIP_FILE_PATH)

        # downloading zip file
        if os.system(curl_cmd) != 0:
            logger.info('Failure downloading zipped file')
            sys.exit(1)
            return

    logger.info('Unzipping the dataset')
    unzip_cmd = 'tar xf {} -C {}'.format(ZIP_FILE_PATH, os.path.dirname(UNZIPPED_FILE_PATH))
    if os.system(unzip_cmd) != 0:
        logger.info('Failure unzipping file')
        sys.exit(1)


def extract_images():
    """
    Extracts images from downloaded dataset. Total 30281 images
    :return: a dictionary :
        - key: an integer
        - value: np array of 86,86,3

    """
    fw_dataset = sio.loadmat(MAT_FILE)
    metadata = fw_dataset['metaData']
    lexicon = fw_dataset['lexicon']

    dirpath = 'faceData/'

    # the actual format is this when reading .mat file
    # imgpath = metadata[0, 0][0][0][0][0]

    imgset = metadata[0]

    # dictionary with key i and numpy image of (ht, wt, channels) as value
    images_dataset = {}

    for i in range(0, len(imgset)):
        imgpath = dirpath + imgset[i][0][0][0][0]
        img = imread(imgpath)
        images_dataset[i] = img
    return images_dataset

def get_images():

    download_and_unzip()
    im = get_images()
    return im





