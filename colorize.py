from __future__ import print_function
import autocolorize as ac
import caffe
import numpy as n
import logging

def configure_logging():
    # Rewrite log
    logging.basicConfig(filename='ac.log',filemode='w',level=logging.DEBUG)

def main():
    IMG_SIZE=512
    net=ac.load_default_classifier(input_size=IMG_SIZE)
    logging.info('Loaded network')
    img=ac.image.load('orig.jpg')
    rgb,info=ac.colorize(img,classifier=net,return_info=True)
    ac.image.save('color_orig.jpg',rgb)


if __name__=='__main__':
    main()
