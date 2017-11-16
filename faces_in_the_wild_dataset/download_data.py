import os
import sys
import logging
import numpy as np
import scipy.io as sio
from scipy.misc import imread
from scipy.misc import imshow
import cPickle as pickle
import config


ZIP_FILE_URL = config.ZIP_FILE_URL
ZIP_FILE_PATH = config.ZIP_FILE_PATH
UNZIPPED_FILE_PATH = config.UNZIPPED_FILE_PATH
MAT_FILE = config.MAT_FILE
DIR_NAME=config.DIR_NAME
IMG_DICT_PICKLE_PATH=config.IMG_DICT_PICKLE_PATH
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
        # print('Download dataset');
        logger.info('Downloading faces-in-the-wild dataset zip file')
        curl_cmd = 'curl {} -o {}'.format(ZIP_FILE_URL, ZIP_FILE_PATH)

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
        logger.info('Unzipped  successfully')


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

    dirpath = UNZIPPED_FILE_PATH+DIR_NAME

    # the actual format is this when reading .mat file
    # imgpath = metadata[0, 0][0][0][0][0]

    imgset = metadata[0]

    # dictionary with key i and numpy image of (ht, wt, channels) as value
    images_dataset = {}
    logging.info('Creating the image dict')
    for i in range(0, len(imgset)):
        imgpath = dirpath + imgset[i][0][0][0][0]
        img = imread(imgpath)
        img=img.astype(np.float64)/255
        images_dataset[i] = img[...,:3].mean(-1)
        if (i+1)%100==0:
            logging.info('Read '+str(i+1)+' images')
    logging.info('Created the image dict')
    return images_dataset


def get_images():
    if os.path.exists(IMG_DICT_PICKLE_PATH):
        logging.info('Pickle file found, loading image dict')
        with open(IMG_DICT_PICKLE_PATH,'rb') as pFile:
            return pickle.load(pFile)
    else:
        logging.info('Pickle file not found')
        download_and_unzip()
        im = extract_images()
        logging.info('Dump image dict into '+IMG_DICT_PICKLE_PATH)
        with open(IMG_DICT_PICKLE_PATH,'wb') as pFile:
            pickle.dump(im, pFile)
    return im





