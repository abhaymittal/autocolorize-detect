import os
import re
import sys
import logging
import numpy as np
import scipy.io as sio
from scipy.misc import imread
from scipy.misc import imshow
import config


class FileReader:

    def __init__(self):

        self.CURRENT_DIR = './'

        self.ZIP_FILE_URL = config.CALTECH101_ZIP_URL
        self.ZIP_FILE_PATH = config.CALTECH101_ZIP_FILE_PATH
        self.UNZIPPED_FILE_PATH = config.CALTECH101_UNZIPPED_FILE_PATH
        self.CALTECH_DIR = self.UNZIPPED_FILE_PATH + '101_ObjectCategories/'
        self.download_and_unzip()

        self.map_category_to_files = {}
        self.list_files = []
        self.categories = []
        self.processedSet = set()
        self.cat_unread_idx={}
        self.read_files()


    def download_and_unzip(self):
         """
         Downlaods the zip file and unzips the dataset.
         :return:
         """

         logger = logging.getLogger()
         if os.path.exists(self.UNZIPPED_FILE_PATH):
             # unzipped path exists
             logger.info('Unzipped dataset exists')
             return

         if not os.path.exists(self.ZIP_FILE_PATH):
             print "downloading"
             # donwload zip file
             # print('Download dataset');
             logger.info('Downloading Caltech  dataset zip file')
             curl_cmd = 'curl {} -o {}'.format(self.ZIP_FILE_URL, self.ZIP_FILE_PATH)

             # downloading zip file
             if os.system(curl_cmd) != 0:
                 logger.info('Failure downloading zipped file')
                 sys.exit(1)
                 return

             logger.info('Unzipping the dataset')
             print "unzipping"
             create_folder = 'mkdir ' + self.UNZIPPED_FILE_PATH
             os.system(create_folder)
             unzip_cmd = 'tar xf {} -C {}'.format(self.ZIP_FILE_PATH, os.path.dirname(self.UNZIPPED_FILE_PATH))
             if os.system(unzip_cmd) != 0:
                 logger.info('Failure unzipping file')
                 sys.exit(1)
             logger.info('Unzipped  successfully')



    def read_files(self):
        print "reading"
        pat = '\w*/(\w*$)'


        list_files = []
        id = 0

        lst = [x[0] for x in os.walk(self.CALTECH_DIR)]
        # The zeroth is current directory
        for i in range(1, len(lst)):
            dir = lst[i]
            match = re.findall(pat, dir)
            # catgory
            category = match[0]
	    self.categories.append(category)

            filenames = [x[2] for x in os.walk(dir)][0]
            filenames = [dir + '/' + file for file in filenames]
            file_ids = range(id, id + len(filenames))
            id += len(filenames)

            self.map_category_to_files[category] = (dir, filenames, file_ids)
            self.cat_unread_idx[category]=0
            self.list_files.extend(filenames)


    def getCategories(self):
        logger = logging.getLogger()
        if self.categories == None or len(self.categories) == 0:
            logger.error("categories for CALTECH101 not initialized")
        return self.categories	




    def getNextFiles(self, category, num_files):
        """

        :param category:
        :param numFiles:
        :return: a list of numpy images, a list of file ids
        """
        dir, filepath, file_ids = self.map_category_to_files[category]
        images = []
        images_ids = []
        # for id in file_ids:
        #     if len(images) >= num_files:
        #         break
        #     if not id in self.processedSet:
        #         imgpath = self.list_files[id]
        #         img = imread(imgpath)
        #         images.append(img)
        #         images_ids.append(id)
        #         self.processedSet.add(id)
        
        beg_idx=self.cat_unread_idx[category]

        end_idx=np.minimum(beg_idx+num_files,len(file_ids))
        self.cat_unread_idx[category]=end_idx
        # will later replace images by a np array
        for idx in np.arange(beg_idx,end_idx):
            img_idx=file_ids[idx]
            imgpath = self.list_files[img_idx]
            img=imread(imgpath)
            img=img.astype(np.float64)/255
            if(len(img.shape)==3):
                img=img[...,:3].mean(-1)
            images.append(img)
            images_ids.append(img_idx)
            
        return (images, images_ids)

    def has_more(self,category):
        dirx, filepath, file_ids = self.map_category_to_files[category]
        idx=self.cat_unread_idx[category]
        if idx>=len(file_ids):
            return False
        return True

    def get_image_path_by_id(self, id):
        return self.list_files[id]

    def get_image_by_id(self, id):
        imgpath = self.list_files[id]
        img = imread(imgpath)
        return img

    def get_category_size(self,category):
        _, _, file_ids = self.map_category_to_files[category]
        return len(file_ids)

    def get_cat_id_range(self,category):
        _, _, file_ids = self.map_category_to_files[category]
        return file_ids[0], file_ids[-1]
        
    def isProcessed(self, id):
        return id in self.processedSet


# fr = FileReader()
# x = fr.getNextFiles('pigeon', 2)
# print fr.isProcessed(7276)
# print x[1]
