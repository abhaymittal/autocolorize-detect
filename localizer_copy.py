import sys
import numpy as np
import os

os.environ["GLOG_minloglevel"] = "2"
import autocolorize as ac
import logging

from scipy.misc import imread
from scipy.misc import imshow
from matplotlib import pyplot as plt

try:
    import cPickle as pickle
except:
    import Pickle as pickle

"""
How to run:
python localizer.py <directory of images> <img_type>

NB: Make sure the directory contains only images and all images of same type
"""


class OcclusionWindow:
    def __init__(self):
        self.scales = [(50, 50)]
        self.strideX = 20
        self.strideY = 20
        self.neural_response = []

        # Loading the autocolorie model

        IMG_SIZE = 576
        self.net = ac.load_default_classifier(input_size=IMG_SIZE)

        # TODO: move to config file

        self.map_imgtype_neuron = {}
        self.map_imgtype_neuron['humans'] = [3867, 4018, 525]
        self.map_imgtype_neuron['airplanes'] = [2590, 4087, 2884, 1628, 2732]

        # Logger
        self.logger = logging.getLogger()


    def get_occlusion_window(self, img, img_type):
        """

        :param net: an objct of neural net
        :param img: a numpy image of dimension (ht, wt)
        :return:
        """

        substrate = np.copy(img)
        ht, wt = img.shape

        # get response for unaltered image

        print "for unaltered image"
        res_org = self._get_fc7_activations(substrate, img_type=img_type)
        iteration = 1
        for win_sizeX, win_sizeY in self.scales:
            for x in range(0, ht, self.strideX):
                x_st, x_en = x, min(x + win_sizeX, ht - 1)
                if (x_en - x_st < win_sizeX / 2.0):
                    continue
                for y in range(0, wt, self.strideY):
                    y_st, y_en = y, min(y + win_sizeY, wt - 1)
                    if (y_en - y_st < win_sizeY / 2.0):
                        continue

                    print "Iteration: ", iteration
                    # print "Running for scale: ", win_sizeX, " ", win_sizeY
                    print "x_st: ", x_st, "  x_en: ", x_en, "  ht: ", x_en - x_st
                    print "y_st: ", y_st, "  y_en: ", y_en, "  wt: ", y_en - y_st

                    iteration += 1

                    cache = np.copy(substrate[x_st:x_en, y_st:y_en])
                    substrate[x_st:x_en, y_st:y_en] = 0
                    response = self._get_fc7_activations(substrate, img_type=img_type)
                    substrate[x_st:x_en, y_st:y_en] = cache
                    self.neural_response.append(((x_st, y_st, x_en, y_en), (win_sizeX, win_sizeY), response))

        best_pos, best_win, best_delta = self._get_max_L2_distance(res_org, self.neural_response)
        print best_pos, best_win, best_delta

        if best_pos != None and best_win != None:
            x_st, y_st, x_en, y_en = best_pos
            substrate[x_st:x_en, y_st:y_en] = 0
            rgb, _ = ac.colorize(img, classifier=self.net, return_info=True)
            rgb[x_st:x_en, y_st:y_en, :] = 0
            return best_pos, best_win, best_delta, rgb
        else:
            print "Nothing best found"
            return [None, None, None, None]

    def process_images_from_directory(self, dir, imgtype):
        """
        *** NOTE: Make sure that all images are of same image type ***
        :param dir: The directory which contains images to be processed
        :param imgtype: the category of image.
        :return:
        """
        if (not os.path.exists(dir)):
            self.logger.error("Directory provided does not exists")
            print "Directory provided does not exists"
            sys.exit(-1)

        results = []
        filenames = [x[2] for x in os.walk(dir)][0]

        for file in filenames:
            filepath = dir + '/' + file
            print "Processing image: ", filepath

            self.logger.info("Processing image: " + filepath)
            img = self._read_image_from_path(filepath)
            best_pos, best_win, best_delta, rgb = self.get_occlusion_window(img, imgtype)

            result = [file, best_pos, best_win, best_delta]
            results.append(result)
            if rgb.any() != None:
                to_save_file = dir + '/' + 'output_' + file
                ac.image.save(to_save_file, rgb)

        to_dump_dir = dir + '/' + 'results.pkl'
        with open(to_dump_dir, 'wb') as f:
            pickle.dump(results, f)

    def _get_fc7_activations(self, img, img_type):
        try:
            rgb, _ = ac.colorize(img, classifier=self.net, return_info=True)
            activations = self.net.blobs['fc7'].data
            activations = activations.reshape(activations.shape[1], activations.shape[2], -1)
            responsible_neuron_act = []

            for neuron in self.map_imgtype_neuron[img_type]:
                act_n = activations[neuron, :, :]

                # Uncomment: If you want to only use max/sum of the returned values
                # responsible_neuron_act = self._get_metric_max(responsible_neuron_act)

                responsible_neuron_act.append(act_n)

            return np.asarray(responsible_neuron_act)
        except:
            print "category not found"
            sys.exit(-1)

    def _read_image_from_path(self, path):
        """
        Reads an image from a file and maps it to 0 and 1
        :param path: the path of the image to be read
        :return:
        """
        if (not os.path.exists(path)):
            self.logger.error(" Image not exists: " + path)
            sys.exit(-1)
        img = imread(path, 'F')
        img = img / 255.0
        if (len(img.shape) == 3):
            img = img[..., :3].mean(-1)
        return img

    def _get_metric_max(self, wts):
        """
        Compares by taking max value of the activation array of a neuron
        :return:
        """
        return np.max(wts)

    def _get_metric_sum(self, wts):
        """
        Compares by summing the activation array of a neuron
        :return:
        """
        return np.sum(wts)

    def _get_max_L2_distance(self, src, candidates):
        print "src shape: ", src.shape
        best_pos, best_win, best_delta = None, None, 0  # np.linalg.norm(src) * 0.3
        for pos, win, response in candidates:
            delta = np.linalg.norm(src - response)
            if delta > best_delta:
                best_pos = pos
                best_win = win
                best_delta = delta

        return best_pos, best_win, best_delta


if len(sys.argv) < 2:
    print "Image directory and type not found"
    sys.exit(-1)

dir, imgtype = sys.argv[1], sys.argv[2]
print dir, imgtype
oc = OcclusionWindow()
oc.process_images_from_directory(dir, imgtype)

