import numpy as np
import os
os.environ["GLOG_minloglevel"] = "2"
import autocolorize as ac


from scipy.misc import imread
from scipy.misc import imshow
from matplotlib import pyplot as plt

#import colorize_max_act as cma


class OcclusionWindow:

    def __init__(self):
        self.scales = [120, 160]
        self.stride = 80
        self.neural_response = []

        # Loading the autocolorie model

        IMG_SIZE = 576
        self.net = ac.load_default_classifier(input_size=IMG_SIZE)

        # TODO: move to config file

        self.map_imgtype_neuron = {}
        self.map_imgtype_neuron['humans'] = [3867, 548, 4056, 4018, 525]
        self.map_imgtype_neuron['airplanes'] = [0, 0]


    def get_fc7_activations(self, img, img_type):
        try :
            rgb, _ = ac.colorize(img, classifier=self.net, return_info=True)
            activations = self.net.blobs['fc7'].data
            #print activations.shape
            activations = activations.reshape(activations.shape[1], activations.shape[2], -1)
            max_act = []
            for neuron in self.map_imgtype_neuron[img_type]:
                act_n = activations[neuron, :, :]
                act_n = np.squeeze(act_n)
                act_n = self.get_metric_sum(act_n)
                max_act.append(act_n)
            
            #print max_act
            return np.asarray(max_act)
        except:
            print "category not found"
        
        


    def get_occlusion_response_multi_channel(self, net, img, image_type):
        """

        :param net: an object of neural net
        :param img: a numpy image of dimension (ht, wt, num_channels)
        :return:
        """

        # we are going to modify substrate
        substrate = np.copy(img)
        ht = img.shape[0]
        wt = img.shape[1]
        for win_size in self.scales:
            for x in range(0, ht, self.stride):
                for y in range(0, wt, self.stride):
                    x_st, x_en = x, min(x + win_size, ht-1)
                    y_st, y_en = y, min(y + win_size, wt-1)
                    cache = substrate[x_st:x_en, y_st:y_en, :]
                    substrate[x_st:x_en, y_st:y_en, :] = 0
                    #response = net.response(substrate, image_type)
                    imshow(substrate)
                    substrate[x_st:x_en, y_st:y_en, :] = cache
                    self.neural_response.append( (x_st, x_en, y_st, y_en, win_size, response) )


    def get_occlusion_response(self):
        """

        :param net: an objct of neural net
        :param img: a numpy image of dimension (ht, wt)
        :return:
        """

	img = self.read_image_from_path('footballer.png')
        substrate = np.copy(img)
        ht, wt = img.shape

        # get response for unaltered image

        res_org = self.get_fc7_activations(substrate, img_type='humans')

        for win_size in self.scales:
            for x in range(0, ht, self.stride):
                x_st, x_en = x, min(x + win_size, ht - 1)
                if (x_en - x_st < win_size / 2.0):
                    continue
                for y in range(0, wt, self.stride):
                    y_st, y_en = y, min(y + win_size, wt-1)
                    if (y_en - y_st < win_size/2.0) :
                        continue
                    print "Running for scale: ", win_size
                    print "x_st: ", x_st, "  x_en: ", x_en, "  ht: ", x_en - x_st
                    print "y_st: ", y_st, "  y_en: ", y_en, "  wt: ", y_en - y_st
                    cache = np.copy(substrate[x_st:x_en, y_st:y_en])
                    substrate[x_st:x_en, y_st:y_en] = 0
                    response = self.get_fc7_activations(substrate, img_type='humans')
                    print "response: ", response
                    substrate[x_st:x_en, y_st:y_en] = cache
                    self.neural_response.append( ( (x_st, y_st, x_en, y_en), win_size, response) )

        best_pos, best_win, best_delta = self.get_max_L2_distance(res_org, self.neural_response)
        print best_pos, best_win, best_delta
        x_st, y_st, x_en, y_en = best_pos
        substrate[x_st:x_en, y_st:y_en] = 0
        ac.image.save('footballer_mod.png', substrate)
        rgb, _ = ac.colorize(img, classifier=self.net, return_info=True)
        rgb[x_st:x_en, y_st:y_en, :] = 0
        ac.image.save('colorized_footballer.png', rgb)

    def read_image_from_path(self, path):
        img = imread(path, 'F')
        img = img/255.0
        if (len(img.shape) == 3):
             img=img[...,:3].mean(-1)
        return img
    

      
    def get_neuron_response(self, net, substrate, image_type):
        activations = cma.get_fc7_activation()


    def get_metric_max(self, wts):
        """
        Compares by taking max value of the activation array of a neuron
        :return:
        """
        return np.max(wts)

    def get_metric_sum(self, wts):
        """
        Compares by summing the activation array of a neuron
        :return:
        """
        return np.sum(wts)

    def get_max_L2_distance(self, src, candidates):
        best_pos, best_win, best_delta = None, None, 10000000
        for pos, win, response in candidates:
            delta = np.linalg.norm(src - response)
            if delta < best_delta:
                best_pos = pos
                best_win = win
                best_delta = delta

        return best_pos, best_win, best_delta




oc = OcclusionWindow()
oc.get_occlusion_response()
