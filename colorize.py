from __future__ import print_function
import autocolorize as ac
import caffe
import numpy as n
import logging
import argparse

def configure_logging(log_level=logging.INFO):
    # Rewrite log
    logging.basicConfig(filename='ac.log',filemode='w',level=log_level)

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('-s','--save_img', 
                        help='True to store colorized images, else false',action='store_true')
    parser.add_argument('-d','--debug',help='Flag to enable debugging',
                        action='store_true')
    return parser.parse_args();

def get_fc7_activations(net):
    fc7=net.blobs['fc7']
    return fc7.data

def main():
    IMG_SIZE=128
    args=parse_args()

    # configure logging
    if args.debug:
        configure_logging(level=logging.DEBUG)
    else:
        configure_logging()

    # load the network and get the images
    net=ac.load_default_classifier(input_size=IMG_SIZE)
    logging.info('Loaded network')
    image_dict=dict();

    # Process the images
    for img_name in image_dict:
        # img=ac.image.load('face.jpg')
        logging.info('Colorizing',img_name)
        img=image_dict[img_name]
        rgb,info=ac.colorize(img,classifier=net,return_info=True)
        activation=get_fc7_activations(net)
        if args.save:
            ac.image.save('color_'+str(img_name)+'.jpg',rgb)


if __name__=='__main__':
    main()
